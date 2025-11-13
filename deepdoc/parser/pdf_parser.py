#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import math
import os
import random
import re
import sys
import threading
from collections import Counter, defaultdict
from copy import deepcopy
from io import BytesIO
import io
from timeit import default_timer as timer
from pathlib import Path

import numpy as np
import pdfplumber
import trio
import xgboost as xgb
from huggingface_hub import snapshot_download
from PIL import Image
# Ensure Image.Resampling exists on older Pillow versions
if not hasattr(Image, "Resampling"):
    class _Resampling:
        LANCZOS = Image.LANCZOS
    Image.Resampling = _Resampling
from pypdf import PdfReader as pdf2_read

from api.utils.file_utils import get_project_base_directory
try:
    from api.utils.misc_utils import pip_install_torch
except Exception:
    def pip_install_torch():
        # Fallback noop if the helper is not available in this codebase.
        return None
from deepdoc.vision import OCR, AscendLayoutRecognizer, LayoutRecognizer, Recognizer, TableStructureRecognizer
from rag.app.picture import vision_llm_chunk, vision_llm_chunk as picture_vision_llm_chunk
from rag.nlp import rag_tokenizer
from rag.prompts.generator import vision_llm_describe_prompt
from rag.settings import PARALLEL_DEVICES

LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
if LOCK_KEY_pdfplumber not in sys.modules:
    sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int = 32, target_max_dimension: int = 1024) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The longest dimension does not exceed 'target_max_dimension'.
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    # Determine the current longest dimension
    max_dimension = max(height, width)
    
    # Calculate scale to fit within target_max_dimension
    scale = target_max_dimension / max_dimension
    
    # Calculate new dimensions while preserving aspect ratio
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    # Ensure both dimensions are divisible by the factor
    adjusted_height = round_by_factor(new_height, factor)
    adjusted_width = round_by_factor(new_width, factor)
    
    # Final check to ensure we don't exceed target_max_dimension after rounding
    if max(adjusted_height, adjusted_width) > target_max_dimension:
        # If rounding pushed us over the limit, reduce slightly
        final_scale = (target_max_dimension - factor) / max(new_height, new_width)
        adjusted_height = round_by_factor(int(height * final_scale), factor)
        adjusted_width = round_by_factor(int(width * final_scale), factor)
    
    # Ensure minimum size is at least the factor to avoid zero or very small dimensions
    adjusted_height = max(adjusted_height, factor)
    adjusted_width = max(adjusted_width, factor)
    
    return adjusted_height, adjusted_width


class RAGFlowPdfParser:
    def __init__(self, vision_model=None, **kwargs):
        """
        If you have trouble downloading HuggingFace models, -_^ this might help!!

        For Linux:
        export HF_ENDPOINT=https://hf-mirror.com

        For Windows:
        Good luck
        ^_-

        Accepts optional vision_model so callers can attach a VLM bundle (LLMBundle or compatible)
        for hybrid table parsing. This guarantees the parser has a consistent attribute whether
        called via different code paths.
        """

        # Attach provided vision model (may be None)
        self.vision_model = vision_model
        logging.debug(f"RAGFlowPdfParser.__init__: vision_model provided: {bool(self.vision_model)}")

        self.ocr = OCR()
        self.parallel_limiter = None
        if PARALLEL_DEVICES > 1:
            self.parallel_limiter = [trio.CapacityLimiter(1) for _ in range(PARALLEL_DEVICES)]

        layout_recognizer_type = os.getenv("LAYOUT_RECOGNIZER_TYPE", "onnx").lower()
        if layout_recognizer_type not in ["onnx", "ascend"]:
            raise RuntimeError("Unsupported layout recognizer type.")

        if hasattr(self, "model_speciess"):
            recognizer_domain = "layout." + self.model_speciess
        else:
            recognizer_domain = "layout"

        if layout_recognizer_type == "ascend":
            logging.debug("Using Ascend LayoutRecognizer")
            self.layouter = AscendLayoutRecognizer(recognizer_domain)
        else:  # onnx
            logging.debug("Using Onnx LayoutRecognizer")
            self.layouter = LayoutRecognizer(recognizer_domain)
        self.tbl_det = TableStructureRecognizer()

        self.updown_cnt_mdl = xgb.Booster()
        try:
            pip_install_torch()
            import torch.cuda
            if torch.cuda.is_available():
                self.updown_cnt_mdl.set_param({"device": "cuda"})
        except Exception:
            logging.info("No torch found.")
        try:
            model_dir = os.path.join(get_project_base_directory(), "rag/res/deepdoc")
            self.updown_cnt_mdl.load_model(os.path.join(model_dir, "updown_concat_xgb.model"))
        except Exception:
            model_dir = snapshot_download(repo_id="InfiniFlow/text_concat_xgb_v1.0", local_dir=os.path.join(get_project_base_directory(), "rag/res/deepdoc"), local_dir_use_symlinks=False)
            self.updown_cnt_mdl.load_model(os.path.join(model_dir, "updown_concat_xgb.model"))

        self.page_from = 0
        self.column_num = 1

    def __char_width(self, c):
        return (c["x1"] - c["x0"]) // max(len(c["text"]), 1)

    def __height(self, c):
        return c["bottom"] - c["top"]

    def _x_dis(self, a, b):
        return min(abs(a["x1"] - b["x0"]), abs(a["x0"] - b["x1"]), abs(a["x0"] + a["x1"] - b["x0"] - b["x1"]) / 2)

    def _y_dis(self, a, b):
        return (b["top"] + b["bottom"] - a["top"] - a["bottom"]) / 2

    def _match_proj(self, b):
        proj_patt = [
            r"第[零一二三四五六七八九十百]+章",
            r"第[零一二三四五六七八九十百]+[条节]",
            r"[零一二三四五六七八九十百]+[、是 　]",
            r"[\(（][零一二三四五六七八九十百]+[）\)]",
            r"[\(（][0-9]+[）\)]",
            r"[0-9]+(、|\.[　 ]|）|\.[^0-9./a-zA-Z_%><-]{4,})",
            r"[0-9]+\.[0-9.]+(、|\.[ 　])",
            r"[⚫•➢①② ]",
        ]
        return any([re.match(p, b["text"]) for p in proj_patt])

    def _updown_concat_features(self, up, down):
        w = max(self.__char_width(up), self.__char_width(down))
        h = max(self.__height(up), self.__height(down))
        y_dis = self._y_dis(up, down)
        LEN = 6
        tks_down = rag_tokenizer.tokenize(down["text"][:LEN]).split()
        tks_up = rag_tokenizer.tokenize(up["text"][-LEN:]).split()
        tks_all = up["text"][-LEN:].strip() + (" " if re.match(r"[a-zA-Z0-9]+", up["text"][-1] + down["text"][0]) else "") + down["text"][:LEN].strip()
        tks_all = rag_tokenizer.tokenize(tks_all).split()
        fea = [
            up.get("R", -1) == down.get("R", -1),
            y_dis / h,
            down["page_number"] - up["page_number"],
            up["layout_type"] == down["layout_type"],
            up["layout_type"] == "text",
            down["layout_type"] == "text",
            up["layout_type"] == "table",
            down["layout_type"] == "table",
            True if re.search(r"([。？！；!?;+)）]|[a-z]\.)$", up["text"]) else False,
            True if re.search(r"[，：‘“、0-9（+-]$", up["text"]) else False,
            True if re.search(r"(^.?[/,?;:\]，。；：’”？！》】）-])", down["text"]) else False,
            True if re.match(r"[\(（][^\(\)（）]+[）\)]$", up["text"]) else False,
            True if re.search(r"[，,][^。.]+$", up["text"]) else False,
            True if re.search(r"[，,][^。.]+$", up["text"]) else False,
            True if re.search(r"[\(（][^\)）]+$", up["text"]) and re.search(r"[\)）]", down["text"]) else False,
            self._match_proj(down),
            True if re.match(r"[A-Z]", down["text"]) else False,
            True if re.match(r"[A-Z]", up["text"][-1]) else False,
            True if re.match(r"[a-z0-9]", up["text"][-1]) else False,
            True if re.match(r"[0-9.%,-]+$", down["text"]) else False,
            up["text"].strip()[-2:] == down["text"].strip()[-2:] if len(up["text"].strip()) > 1 and len(down["text"].strip()) > 1 else False,
            up["x0"] > down["x1"],
            abs(self.__height(up) - self.__height(down)) / min(self.__height(up), self.__height(down)),
            self._x_dis(up, down) / max(w, 0.000001),
            (len(up["text"]) - len(down["text"])) / max(len(up["text"]), len(down["text"])),
            len(tks_all) - len(tks_up) - len(tks_down),
            len(tks_down) - len(tks_up),
            tks_down[-1] == tks_up[-1] if tks_down and tks_up else False,
            max(down["in_row"], up["in_row"]),
            abs(down["in_row"] - up["in_row"]),
            len(tks_down) == 1 and rag_tokenizer.tag(tks_down[0]).find("n") >= 0,
            len(tks_up) == 1 and rag_tokenizer.tag(tks_up[0]).find("n") >= 0,
        ]
        return fea

    @staticmethod
    def sort_X_by_page(arr, threshold):
        # sort using y1 first and then x1
        arr = sorted(arr, key=lambda r: (r["page_number"], r["x0"], r["top"]))
        for i in range(len(arr) - 1):
            for j in range(i, -1, -1):
                # restore the order using th
                if abs(arr[j + 1]["x0"] - arr[j]["x0"]) < threshold and arr[j + 1]["top"] < arr[j]["top"] and arr[j + 1]["page_number"] == arr[j]["page_number"]:
                    tmp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = tmp
        return arr

    def _has_color(self, o):
        if o.get("ncs", "") == "DeviceGray":
            if o["stroking_color"] and o["stroking_color"][0] == 1 and o["non_stroking_color"] and o["non_stroking_color"][0] == 1:
                if re.match(r"[a-zT_\[\]\(\)-]+", o.get("text", "")):
                    return False
        return True

    def _table_transformer_job(self, ZM):
        # Add validation to confirm self.vision_model exists before attempting VLM parsing
        vision_present = hasattr(self, "vision_model") and getattr(self, "vision_model", None) is not None
        logging.info(f"VLM table parsing: vision_model present={vision_present}")
        logging.debug("Table processing...")
        imgs, pos = [], []
        tbcnt = [0]
        MARGIN = 10
        self.tb_cpns = []
        assert len(self.page_layout) == len(self.page_images)
        logging.info(f"_table_transformer_job: Extracting table regions from {len(self.page_layout)} pages")
        for p, tbls in enumerate(self.page_layout):  # for page
            tbls = [f for f in tbls if f["type"] == "table"]
            tbcnt.append(len(tbls))
            if not tbls:
                continue
            logging.info(f"_table_transformer_job: Page {p+1} has {len(tbls)} table regions")
            for tb in tbls:  # for table
                left, top, right, bott = tb["x0"] - MARGIN, tb["top"] - MARGIN, tb["x1"] + MARGIN, tb["bottom"] + MARGIN
                left *= ZM
                top *= ZM
                right *= ZM
                bott *= ZM
                pos.append((left, top))
                imgs.append(self.page_images[p].crop((left, top, right, bott)))
                logging.debug(f"_table_transformer_job: Extracted table region - page={p+1}, bbox=({tb['x0']:.1f},{tb['top']:.1f},{tb['x1']:.1f},{tb['bottom']:.1f}), crop_coords=({left:.1f},{top:.1f},{right:.1f},{bott:.1f})")

        assert len(self.page_images) == len(tbcnt) - 1
        if not imgs:
            logging.info("_table_transformer_job: No table images to process")
            return
        
        # Step 2: Optional VLM routing for tables (hybrid VLM/TableStructureRecognizer)
        env_flag = os.getenv("USE_VLM_TABLE_PARSING", "false")
        vlm_enabled = str(env_flag).lower() == "true" and vision_present
        
        # Add explicit logging of environment variable values
        logging.info(f"USE_VLM_TABLE_PARSING={env_flag}, vision_model_present={vision_present}, vlm_enabled={vlm_enabled}")
        
        logging.info(f"Table parsing path: VLM={vlm_enabled}, table_count={len(imgs)}")
        
        # Add validation that required environment variables are set before attempting VLM parsing
        if vlm_enabled and not vision_present:
            logging.warning("_table_transformer_job: VLM parsing requested but no vision_model available; falling back to TableStructureRecognizer")
            vlm_enabled = False
        
        # Verify that VLM_TABLE_MODEL environment variable is properly used when creating the vision model
        if vlm_enabled:
            # Check if VLM_TABLE_MODEL is set and log it
            table_model_env = os.getenv("VLM_TABLE_MODEL", None)
            if table_model_env:
                logging.info(f"VLM_TABLE_MODEL environment variable is set to: {table_model_env}")
            else:
                logging.info("VLM_TABLE_MODEL environment variable is not set, using default vision_model")
        
        recos = None
        if vlm_enabled:
            # Allow a separate model for table parsing if provided (env override preferred)
            table_model_env = os.getenv("VLM_TABLE_MODEL", None)
            table_model = table_model_env if table_model_env is not None else self.vision_model
            logging.debug(f"_table_transformer_job: selected table_model from env='{table_model_env}' type={type(table_model)}")
            try:
                vlm_results = self._vlm_table_parser(imgs, pos, vision_model=table_model)
                logging.debug(f"_table_transformer_job: vlm_results received: {len(vlm_results)} items")
            except Exception:
                logging.exception("VLM table parser failed; falling back to TableStructureRecognizer for all tables")
                vlm_results = [None] * len(imgs)
        
            # vlm_results: list of strings (validated html/md) or None (signal fallback)
            # Prepare combined recos: for vlm-success -> [{'html': <str>}] ; for None -> use table structure recognizer
            indices_need_fallback = [i for i, r in enumerate(vlm_results) if r is None]
            logging.info(f"_table_transformer_job: VLM results - successful={len(vlm_results) - len(indices_need_fallback)}, need_fallback={len(indices_need_fallback)}")
            logging.debug(f"_table_transformer_job: indices_need_fallback={indices_need_fallback}")
            fallback_recos = []
            if indices_need_fallback:
                try:
                    # Run TableStructureRecognizer only on tables that need fallback
                    fallback_imgs = [imgs[i] for i in indices_need_fallback]
                    fallback_recos = self.tbl_det(fallback_imgs)
                    logging.debug(f"_table_transformer_job: fallback_recos obtained for {len(fallback_recos)} tables")
                except Exception:
                    logging.exception("TableStructureRecognizer fallback failed for some tables")
                    # ensure we have placeholders so lengths align
                    fallback_recos = [[] for _ in indices_need_fallback]
        
            # Build recos as a list aligned with imgs
            recos = []
            fallback_iter = iter(fallback_recos)
            for i, r in enumerate(vlm_results):
                if r is None:
                    # Insert the next result from the fallback recos (which is a list of component dicts)
                    recos.append(next(fallback_iter, []))
                    logging.debug(f"_table_transformer_job: table {i} using fallback recognition")
                else:
                    # VLM produced an HTML/markdown result string -> wrap as a single dict entry
                    recos.append([{"html": r}])
                    logging.debug(f"_table_transformer_job: table {i} processed with VLM")
        else:
            logging.info("_table_transformer_job: Using TableStructureRecognizer for all tables")
            recos = self.tbl_det(imgs)

        # Step 3: Process recos (can contain VLM-wrapped results or standard TableStructureRecognizer output)
        tbcnt = np.cumsum(tbcnt)
        for i in range(len(tbcnt) - 1):  # for page
            pg = []
            for j, tb_items in enumerate(recos[tbcnt[i] : tbcnt[i + 1]]):  # for table
                poss = pos[tbcnt[i] : tbcnt[i + 1]]
                # If VLM returned table HTML/markdown, tb_items will be a list whose first item is a dict with 'html'
                if tb_items and isinstance(tb_items[0], dict) and "html" in tb_items[0]:
                    # mark VLM result and attach minimal metadata
                    it = tb_items[0]
                    it["source"] = "vlm"
                    it["pn"] = i
                    it["layoutno"] = j
                    # Try to attach approximate bounding box from cropping info if available
                    try:
                        left, top = poss[j]
                        it.setdefault("x0", left / ZM)
                        it.setdefault("x1", (left + imgs[tbcnt[i] + j].size[0]) / ZM)
                        it.setdefault("top", top / ZM + self.page_cum_height[i])
                        it.setdefault("bottom", (top + imgs[tbcnt[i] + j].size[1]) / ZM + self.page_cum_height[i])
                    except Exception:
                        # If pos not available or malformed, leave geometry absent — gather logic will skip VLM items
                        logging.debug("VLM table item missing position metadata")
                    pg.append(it)
                    continue

                # Standard TableStructureRecognizer output: adjust coordinates back to document scale
                for it in tb_items:  # for table components
                    it["x0"] = it["x0"] + poss[j][0]
                    it["x1"] = it["x1"] + poss[j][0]
                    it["top"] = it["top"] + poss[j][1]
                    it["bottom"] = it["bottom"] + poss[j][1]
                    for n in ["x0", "x1", "top", "bottom"]:
                        it[n] /= ZM
                    it["top"] += self.page_cum_height[i]
                    it["bottom"] += self.page_cum_height[i]
                    it["pn"] = i
                    it["layoutno"] = j
                    pg.append(it)
            self.tb_cpns.extend(pg)

        def gather(kwd, fzy=10, ption=0.6):
            eles = Recognizer.sort_Y_firstly([r for r in self.tb_cpns if "label" in r and re.match(kwd, r["label"]) and all(key in r for key in ["top", "x0"])], fzy)
            eles = Recognizer.layouts_cleanup(self.boxes, eles, 5, ption)
            return Recognizer.sort_Y_firstly(eles, 0)

        # add R,H,C,SP tag to boxes within table layout
        headers = gather(r".*header$")
        rows = gather(r".* (row|header)")
        spans = gather(r".*spanning")
        clmns = sorted([r for r in self.tb_cpns if "label" in r and re.match(r"table column$", r["label"]) and all(key in r for key in ["pn", "layoutno", "x0"])], key=lambda x: (x["pn"], x["layoutno"], x["x0"]))
        clmns = Recognizer.layouts_cleanup(self.boxes, clmns, 5, 0.5)
        for b in self.boxes:
            if b.get("layout_type", "") != "table":
                continue
            ii = Recognizer.find_overlapped_with_threshold(b, rows, thr=0.3)
            if ii is not None and "top" in rows[ii] and "bottom" in rows[ii]:
                b["R"] = ii
                b["R_top"] = rows[ii]["top"]
                b["R_bott"] = rows[ii]["bottom"]

            ii = Recognizer.find_overlapped_with_threshold(b, headers, thr=0.3)
            if ii is not None and all(key in headers[ii] for key in ["top", "bottom", "x0", "x1"]):
                b["H_top"] = headers[ii]["top"]
                b["H_bott"] = headers[ii]["bottom"]
                b["H_left"] = headers[ii]["x0"]
                b["H_right"] = headers[ii]["x1"]
                b["H"] = ii

            ii = Recognizer.find_horizontally_tightest_fit(b, clmns)
            if ii is not None and "x0" in clmns[ii] and "x1" in clmns[ii]:
                b["C"] = ii
                b["C_left"] = clmns[ii]["x0"]
                b["C_right"] = clmns[ii]["x1"]

            ii = Recognizer.find_overlapped_with_threshold(b, spans, thr=0.3)
            if ii is not None and all(key in spans[ii] for key in ["top", "bottom", "x0", "x1"]):
                b["H_top"] = spans[ii]["top"]
                b["H_bott"] = spans[ii]["bottom"]
                b["H_left"] = spans[ii]["x0"]
                b["H_right"] = spans[ii]["x1"]
                b["SP"] = ii

    def __ocr(self, pagenum, img, chars, ZM=3, device_id: int | None = None):
        start = timer()
        bxs = self.ocr.detect(np.array(img), device_id)
        logging.info(f"__ocr detecting boxes of a image cost ({timer() - start}s)")

        start = timer()
        if not bxs:
            self.boxes.append([])
            return
        bxs = [(line[0], line[1][0]) for line in bxs]
        bxs = Recognizer.sort_Y_firstly(
            [
                {"x0": b[0][0] / ZM, "x1": b[1][0] / ZM, "top": b[0][1] / ZM, "text": "", "txt": t, "bottom": b[-1][1] / ZM, "chars": [], "page_number": pagenum}
                for b, t in bxs
                if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]
            ],
            self.mean_height[pagenum - 1] / 3,
        )

        # merge chars in the same rect
        for c in chars:
            ii = Recognizer.find_overlapped(c, bxs)
            if ii is None:
                self.lefted_chars.append(c)
                continue
            ch = c["bottom"] - c["top"]
            bh = bxs[ii]["bottom"] - bxs[ii]["top"]
            if abs(ch - bh) / max(ch, bh) >= 0.7 and c["text"] != " ":
                self.lefted_chars.append(c)
                continue
            bxs[ii]["chars"].append(c)

        for b in bxs:
            if not b["chars"]:
                del b["chars"]
                continue
            m_ht = np.mean([c["height"] for c in b["chars"]])
            for c in Recognizer.sort_Y_firstly(b["chars"], m_ht):
                if c["text"] == " " and b["text"]:
                    if re.match(r"[0-9a-zA-Zа-яА-Я,.?;:!%%]", b["text"][-1]):
                        b["text"] += " "
                else:
                    b["text"] += c["text"]
            del b["chars"]

        logging.info(f"__ocr sorting {len(chars)} chars cost {timer() - start}s")
        start = timer()
        boxes_to_reg = []
        img_np = np.array(img)
        for b in bxs:
            if not b["text"]:
                left, right, top, bott = b["x0"] * ZM, b["x1"] * ZM, b["top"] * ZM, b["bottom"] * ZM
                b["box_image"] = self.ocr.get_rotate_crop_image(img_np, np.array([[left, top], [right, top], [right, bott], [left, bott]], dtype=np.float32))
                boxes_to_reg.append(b)
            del b["txt"]
        texts = self.ocr.recognize_batch([b["box_image"] for b in boxes_to_reg], device_id)
        for i in range(len(boxes_to_reg)):
            boxes_to_reg[i]["text"] = texts[i]
            del boxes_to_reg[i]["box_image"]
        logging.info(f"__ocr recognize {len(bxs)} boxes cost {timer() - start}s")
        bxs = [b for b in bxs if b["text"]]
        if self.mean_height[pagenum - 1] == 0:
            self.mean_height[pagenum - 1] = np.median([b["bottom"] - b["top"] for b in bxs])
        self.boxes.append(bxs)

    def _layouts_rec(self, ZM, drop=True):
        assert len(self.page_images) == len(self.boxes)
        self.boxes, self.page_layout = self.layouter(self.page_images, self.boxes, ZM, drop=drop)
        # cumlative Y
        for i in range(len(self.boxes)):
            self.boxes[i]["top"] += self.page_cum_height[self.boxes[i]["page_number"] - 1]
            self.boxes[i]["bottom"] += self.page_cum_height[self.boxes[i]["page_number"] - 1]

    def _assign_column(self, boxes, zoomin=3):
        if not boxes:
            return boxes

        if all("col_id" in b for b in boxes):
            return boxes

        by_page = defaultdict(list)
        for b in boxes:
            by_page[b["page_number"]].append(b)

        page_info = {}  # pg -> dict(page_w, left_edge, cand_cols)
        counter = Counter()

        for pg, bxs in by_page.items():
            if not bxs:
                page_info[pg] = {"page_w": 1.0, "left_edge": 0.0, "cand": 1}
                counter[1] += 1
                continue

            if hasattr(self, "page_images") and self.page_images and len(self.page_images) >= pg:
                page_w = self.page_images[pg - 1].size[0] / max(1, zoomin)
                left_edge = 0.0
            else:
                xs0 = [box["x0"] for box in bxs]
                xs1 = [box["x1"] for box in bxs]
                left_edge = float(min(xs0))
                page_w = max(1.0, float(max(xs1) - left_edge))

            widths = [max(1.0, (box["x1"] - box["x0"])) for box in bxs]
            median_w = float(np.median(widths)) if widths else 1.0

            raw_cols = int(page_w / max(1.0, median_w))

            # cand = raw_cols if (raw_cols >= 2 and median_w < page_w / raw_cols * 0.8) else 1
            cand = raw_cols

            page_info[pg] = {"page_w": page_w, "left_edge": left_edge, "cand": cand}
            counter[cand] += 1

            logging.info(f"[Page {pg}] median_w={median_w:.2f}, page_w={page_w:.2f}, raw_cols={raw_cols}, cand={cand}")

        global_cols = counter.most_common(1)[0][0]
        logging.info(f"Global column_num decided by majority: {global_cols}")

        for pg, bxs in by_page.items():
            if not bxs:
                continue

            page_w = page_info[pg]["page_w"]
            left_edge = page_info[pg]["left_edge"]

            if global_cols == 1:
                for box in bxs:
                    box["col_id"] = 0
                continue

            for box in bxs:
                w = box["x1"] - box["x0"]
                if w >= 0.8 * page_w:
                    box["col_id"] = 0
                    continue
                cx = 0.5 * (box["x0"] + box["x1"])
                norm_cx = (cx - left_edge) / page_w
                norm_cx = max(0.0, min(norm_cx, 0.999999))
                box["col_id"] = int(min(global_cols - 1, norm_cx * global_cols))

        return boxes

    def _text_merge(self, zoomin=3):
        # merge adjusted boxes
        bxs = self._assign_column(self.boxes, zoomin)

        def end_with(b, txt):
            txt = txt.strip()
            tt = b.get("text", "").strip()
            return tt and tt.find(txt) == len(tt) - len(txt)

        def start_with(b, txts):
            tt = b.get("text", "").strip()
            return tt and any([tt.find(t.strip()) == 0 for t in txts])

        # horizontally merge adjacent box with the same layout
        i = 0
        while i < len(bxs) - 1:
            b = bxs[i]
            b_ = bxs[i + 1]

            if b["page_number"] != b_["page_number"] or b.get("col_id") != b_.get("col_id"):
                i += 1
                continue

            if b.get("layoutno", "0") != b_.get("layoutno", "1") or b.get("layout_type", "") in ["table", "figure", "equation"]:
                i += 1
                continue

            if abs(self._y_dis(b, b_)) < self.mean_height[bxs[i]["page_number"] - 1] / 3:
                # merge
                bxs[i]["x1"] = b_["x1"]
                bxs[i]["top"] = (b["top"] + b_["top"]) / 2
                bxs[i]["bottom"] = (b["bottom"] + b_["bottom"]) / 2
                bxs[i]["text"] += b_["text"]
                bxs.pop(i + 1)
                continue
            i += 1
        self.boxes = bxs

    def _naive_vertical_merge(self, zoomin=3):
        bxs = self._assign_column(self.boxes, zoomin)

        grouped = defaultdict(list)
        for b in bxs:
            grouped[(b["page_number"], b.get("col_id", 0))].append(b)

        merged_boxes = []
        for (pg, col), bxs in grouped.items():
            bxs = sorted(bxs, key=lambda x: (x["top"], x["x0"]))
            if not bxs:
                continue

            mh = self.mean_height[pg - 1] if self.mean_height else np.median([b["bottom"] - b["top"] for b in bxs]) or 10

            i = 0
            while i + 1 < len(bxs):
                b = bxs[i]
                b_ = bxs[i + 1]

                if b["page_number"] < b_["page_number"] and re.match(r"[0-9  •一—-]+$", b["text"]):
                    bxs.pop(i)
                    continue

                if not b["text"].strip():
                    bxs.pop(i)
                    continue

                if not b["text"].strip() or b.get("layoutno") != b_.get("layoutno"):
                    i += 1
                    continue

                if b_["top"] - b["bottom"] > mh * 1.5:
                    i += 1
                    continue

                overlap = max(0, min(b["x1"], b_["x1"]) - max(b["x0"], b_["x0"]))
                if overlap / max(1, min(b["x1"] - b["x0"], b_["x1"] - b_["x0"])) < 0.3:
                    i += 1
                    continue

                concatting_feats = [
                    b["text"].strip()[-1] in ",;:'\"，、‘“；：-",
                    len(b["text"].strip()) > 1 and b["text"].strip()[-2] in ",;:'\"，‘“、；：",
                    b_["text"].strip() and b_["text"].strip()[0] in "。；？！?”）),，、：",
                ]
                # features for not concating
                feats = [
                    b.get("layoutno", 0) != b_.get("layoutno", 0),
                    b["text"].strip()[-1] in "。？！?",
                    self.is_english and b["text"].strip()[-1] in ".!?",
                    b["page_number"] == b_["page_number"] and b_["top"] - b["bottom"] > self.mean_height[b["page_number"] - 1] * 1.5,
                    b["page_number"] < b_["page_number"] and abs(b["x0"] - b_["x0"]) > self.mean_width[b["page_number"] - 1] * 4,
                ]
                # split features
                detach_feats = [b["x1"] < b_["x0"], b["x0"] > b_["x1"]]
                if (any(feats) and not any(concatting_feats)) or any(detach_feats):
                    logging.debug(
                        "{} {} {} {}".format(
                            b["text"],
                            b_["text"],
                            any(feats),
                            any(concatting_feats),
                        )
                    )
                    i += 1
                    continue

                b["text"] = (b["text"].rstrip() + " " + b_["text"].lstrip()).strip()
                b["bottom"] = b_["bottom"]
                b["x0"] = min(b["x0"], b_["x0"])
                b["x1"] = max(b["x1"], b_["x1"])
                bxs.pop(i + 1)

            merged_boxes.extend(bxs)

        self.boxes = sorted(merged_boxes, key=lambda x: (x["page_number"], x.get("col_id", 0), x["top"]))

    def _final_reading_order_merge(self, zoomin=3):
        if not self.boxes:
            return

        self.boxes = self._assign_column(self.boxes, zoomin=zoomin)

        pages = defaultdict(lambda: defaultdict(list))
        for b in self.boxes:
            pg = b["page_number"]
            col = b.get("col_id", 0)
            pages[pg][col].append(b)

        for pg in pages:
            for col in pages[pg]:
                pages[pg][col].sort(key=lambda x: (x["top"], x["x0"]))

        new_boxes = []
        for pg in sorted(pages.keys()):
            for col in sorted(pages[pg].keys()):
                new_boxes.extend(pages[pg][col])

        self.boxes = new_boxes

    def _concat_downward(self, concat_between_pages=True):
        self.boxes = Recognizer.sort_Y_firstly(self.boxes, 0)
        return

        # count boxes in the same row as a feature
        for i in range(len(self.boxes)):
            mh = self.mean_height[self.boxes[i]["page_number"] - 1]
            self.boxes[i]["in_row"] = 0
            j = max(0, i - 12)
            while j < min(i + 12, len(self.boxes)):
                if j == i:
                    j += 1
                    continue
                ydis = self._y_dis(self.boxes[i], self.boxes[j]) / mh
                if abs(ydis) < 1:
                    self.boxes[i]["in_row"] += 1
                elif ydis > 0:
                    break
                j += 1

        # concat between rows
        boxes = deepcopy(self.boxes)
        blocks = []
        while boxes:
            chunks = []

            def dfs(up, dp):
                chunks.append(up)
                i = dp
                while i < min(dp + 12, len(boxes)):
                    ydis = self._y_dis(up, boxes[i])
                    smpg = up["page_number"] == boxes[i]["page_number"]
                    mh = self.mean_height[up["page_number"] - 1]
                    mw = self.mean_width[up["page_number"] - 1]
                    if smpg and ydis > mh * 4:
                        break
                    if not smpg and ydis > mh * 16:
                        break
                    down = boxes[i]
                    if not concat_between_pages and down["page_number"] > up["page_number"]:
                        break

                    if up.get("R", "") != down.get("R", "") and up["text"][-1] != "，":
                        i += 1
                        continue

                    if re.match(r"[0-9]{2,3}/[0-9]{3}$", up["text"]) or re.match(r"[0-9]{2,3}/[0-9]{3}$", down["text"]) or not down["text"].strip():
                        i += 1
                        continue

                    if not down["text"].strip() or not up["text"].strip():
                        i += 1
                        continue

                    if up["x1"] < down["x0"] - 10 * mw or up["x0"] > down["x1"] + 10 * mw:
                        i += 1
                        continue

                    if i - dp < 5 and up.get("layout_type") == "text":
                        if up.get("layoutno", "1") == down.get("layoutno", "2"):
                            dfs(down, i + 1)
                            boxes.pop(i)
                            return
                        i += 1
                        continue

                    fea = self._updown_concat_features(up, down)
                    if self.updown_cnt_mdl.predict(xgb.DMatrix([fea]))[0] <= 0.5:
                        i += 1
                        continue
                    dfs(down, i + 1)
                    boxes.pop(i)
                    return

            dfs(boxes[0], 1)
            boxes.pop(0)
            if chunks:
                blocks.append(chunks)

        # concat within each block
        boxes = []
        for b in blocks:
            if len(b) == 1:
                boxes.append(b[0])
                continue
            t = b[0]
            for c in b[1:]:
                t["text"] = t["text"].strip()
                c["text"] = c["text"].strip()
                if not c["text"]:
                    continue
                if t["text"] and re.match(r"[0-9\.a-zA-Z]+$", t["text"][-1] + c["text"][-1]):
                    t["text"] += " "
                t["text"] += c["text"]
                t["x0"] = min(t["x0"], c["x0"])
                t["x1"] = max(t["x1"], c["x1"])
                t["page_number"] = min(t["page_number"], c["page_number"])
                t["bottom"] = c["bottom"]
                if not t["layout_type"] and c["layout_type"]:
                    t["layout_type"] = c["layout_type"]
            boxes.append(t)

        self.boxes = Recognizer.sort_Y_firstly(boxes, 0)

    def _filter_forpages(self):
        if not self.boxes:
            return
        findit = False
        i = 0
        while i < len(self.boxes):
            if not re.match(r"(contents|目录|目次|table of contents|致谢|acknowledge)$", re.sub(r"( | |\u3000)+", "", self.boxes[i]["text"].lower())):
                i += 1
                continue
            findit = True
            eng = re.match(r"[0-9a-zA-Z :'.-]{5,}", self.boxes[i]["text"].strip())
            self.boxes.pop(i)
            if i >= len(self.boxes):
                break
            prefix = self.boxes[i]["text"].strip()[:3] if not eng else " ".join(self.boxes[i]["text"].strip().split()[:2])
            while not prefix:
                self.boxes.pop(i)
                if i >= len(self.boxes):
                    break
                prefix = self.boxes[i]["text"].strip()[:3] if not eng else " ".join(self.boxes[i]["text"].strip().split()[:2])
            self.boxes.pop(i)
            if i >= len(self.boxes) or not prefix:
                break
            for j in range(i, min(i + 128, len(self.boxes))):
                if not re.match(prefix, self.boxes[j]["text"]):
                    continue
                for k in range(i, j):
                    self.boxes.pop(i)
                break
        if findit:
            return

        page_dirty = [0] * len(self.page_images)
        for b in self.boxes:
            if re.search(r"(··|··|··)", b["text"]):
                page_dirty[b["page_number"] - 1] += 1
        page_dirty = set([i + 1 for i, t in enumerate(page_dirty) if t > 3])
        if not page_dirty:
            return
        i = 0
        while i < len(self.boxes):
            if self.boxes[i]["page_number"] in page_dirty:
                self.boxes.pop(i)
                continue
            i += 1

    def _merge_with_same_bullet(self):
        i = 0
        while i + 1 < len(self.boxes):
            b = self.boxes[i]
            b_ = self.boxes[i + 1]
            if not b["text"].strip():
                self.boxes.pop(i)
                continue
            if not b_["text"].strip():
                self.boxes.pop(i + 1)
                continue

            if (
                b["text"].strip()[0] != b_["text"].strip()[0]
                or b["text"].strip()[0].lower() in set("qwertyuopasdfghjklzxcvbnm")
                or rag_tokenizer.is_chinese(b["text"].strip()[0])
                or b["top"] > b_["bottom"]
            ):
                i += 1
                continue
            b_["text"] = b["text"] + "\n" + b_["text"]
            b_["x0"] = min(b["x0"], b_["x0"])
            b_["x1"] = max(b["x1"], b_["x1"])
            b_["top"] = b["top"]
            self.boxes.pop(i)

    def _extract_table_figure(self, need_image, ZM, return_html, need_position, separate_tables_figures=False):
        tables = {}
        figures = {}
        # extract figure and table boxes
        i = 0
        lst_lout_no = ""
        nomerge_lout_no = []
        while i < len(self.boxes):
            if "layoutno" not in self.boxes[i]:
                i += 1
                continue
            lout_no = str(self.boxes[i]["page_number"]) + "-" + str(self.boxes[i]["layoutno"])
            if TableStructureRecognizer.is_caption(self.boxes[i]) or self.boxes[i]["layout_type"] in ["table caption", "title", "figure caption", "reference"]:
                nomerge_lout_no.append(lst_lout_no)
            if self.boxes[i]["layout_type"] == "table":
                if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                if lout_no not in tables:
                    tables[lout_no] = []
                tables[lout_no].append(self.boxes[i])
                self.boxes.pop(i)
                lst_lout_no = lout_no
                continue
            if need_image and self.boxes[i]["layout_type"] == "figure":
                if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                if lout_no not in figures:
                    figures[lout_no] = []
                figures[lout_no].append(self.boxes[i])
                self.boxes.pop(i)
                lst_lout_no = lout_no
                continue
            i += 1

        # merge table on different pages
        nomerge_lout_no = set(nomerge_lout_no)
        tbls = sorted([(k, bxs) for k, bxs in tables.items()], key=lambda x: (x[1][0]["top"], x[1][0]["x0"]))

        i = len(tbls) - 1
        while i - 1 >= 0:
            k0, bxs0 = tbls[i - 1]
            k, bxs = tbls[i]
            i -= 1
            if k0 in nomerge_lout_no:
                continue
            if bxs[0]["page_number"] == bxs0[0]["page_number"]:
                continue
            if bxs[0]["page_number"] - bxs0[0]["page_number"] > 1:
                continue
            mh = self.mean_height[bxs[0]["page_number"] - 1]
            if self._y_dis(bxs0[-1], bxs[0]) > mh * 23:
                continue
            tables[k0].extend(tables[k])
            del tables[k]

        def x_overlapped(a, b):
            return not any([a["x1"] < b["x0"], a["x0"] > b["x1"]])

        # find captions and pop out
        i = 0
        while i < len(self.boxes):
            c = self.boxes[i]
            # mh = self.mean_height[c["page_number"]-1]
            if not TableStructureRecognizer.is_caption(c):
                i += 1
                continue

            # find the nearest layouts
            def nearest(tbls):
                nonlocal c
                mink = ""
                minv = 1000000000
                for k, bxs in tbls.items():
                    for b in bxs:
                        if b.get("layout_type", "").find("caption") >= 0:
                            continue
                        y_dis = self._y_dis(c, b)
                        x_dis = self._x_dis(c, b) if not x_overlapped(c, b) else 0
                        dis = y_dis * y_dis + x_dis * x_dis
                        if dis < minv:
                            mink = k
                            minv = dis
                return mink, minv

            tk, tv = nearest(tables)
            fk, fv = nearest(figures)
            # if min(tv, fv) > 2000:
            #    i += 1
            #    continue
            if tv < fv and tk:
                tables[tk].insert(0, c)
                logging.debug("TABLE:" + self.boxes[i]["text"] + "; Cap: " + tk)
            elif fk:
                figures[fk].insert(0, c)
                logging.debug("FIGURE:" + self.boxes[i]["text"] + "; Cap: " + tk)
            self.boxes.pop(i)

        def cropout(bxs, ltype, poss):
            nonlocal ZM
            pn = set([b["page_number"] - 1 for b in bxs])
            if len(pn) < 2:
                pn = list(pn)[0]
                ht = self.page_cum_height[pn]
                b = {"x0": np.min([b["x0"] for b in bxs]), "top": np.min([b["top"] for b in bxs]) - ht, "x1": np.max([b["x1"] for b in bxs]), "bottom": np.max([b["bottom"] for b in bxs]) - ht}
                louts = [layout for layout in self.page_layout[pn] if layout["type"] == ltype]
                ii = Recognizer.find_overlapped(b, louts, naive=True)
                if ii is not None:
                    b = louts[ii]
                else:
                    logging.warning(f"Missing layout match: {pn + 1},%s" % (bxs[0].get("layoutno", "")))

                left, top, right, bott = b["x0"], b["top"], b["x1"], b["bottom"]
                if right < left:
                    right = left + 1
                poss.append((pn + self.page_from, left, right, top, bott))
                return self.page_images[pn].crop((left * ZM, top * ZM, right * ZM, bott * ZM))
            pn = {}
            for b in bxs:
                p = b["page_number"] - 1
                if p not in pn:
                    pn[p] = []
                pn[p].append(b)
            pn = sorted(pn.items(), key=lambda x: x[0])
            imgs = [cropout(arr, ltype, poss) for p, arr in pn]
            pic = Image.new("RGB", (int(np.max([i.size[0] for i in imgs])), int(np.sum([m.size[1] for m in imgs]))), (245, 245, 245))
            height = 0
            for img in imgs:
                pic.paste(img, (0, int(height)))
                height += img.size[1]
            return pic

        res = []
        positions = []
        figure_results = []
        figure_positions = []
        # crop figure out and add caption
        for k, bxs in figures.items():
            txt = "\n".join([b["text"] for b in bxs])
            if not txt:
                continue

            poss = []

            if separate_tables_figures:
                figure_results.append((cropout(bxs, "figure", poss), [txt]))
                figure_positions.append(poss)
            else:
                res.append((cropout(bxs, "figure", poss), [txt]))
                positions.append(poss)

        for k, bxs in tables.items():
            if not bxs:
                continue
            bxs = Recognizer.sort_Y_firstly(bxs, np.mean([(b["bottom"] - b["top"]) / 2 for b in bxs]))

            poss = []

            # Detect VLM-sourced table items (hybrid path)
            is_vlm_table = any([isinstance(b, dict) and b.get("source") == "vlm" for b in bxs])
            if is_vlm_table:
                # Extract HTML directly from VLM output dicts (prefer first non-empty)
                html = None
                for b in bxs:
                    if isinstance(b, dict) and b.get("html"):
                        html = b.get("html")
                        break

                # If no html was found, fall back to standard path
                if not html:
                    res.append((cropout(bxs, "table", poss), self.tbl_det.construct_table(bxs, html=return_html, is_english=self.is_english)))
                    positions.append(poss)
                    continue

                # Validate / clean HTML table if helper is available
                try:
                    html = self._validate_html_table(html)
                except Exception:
                    pass

                # Try to extract caption text from boxes that are marked as captions
                caption_text = None
                for b in bxs:
                    try:
                        if TableStructureRecognizer.is_caption(b):
                            caption_text = b.get("text", "").strip()
                            if caption_text:
                                break
                    except Exception:
                        continue

                # Insert caption into HTML if not already present and if we found caption text
                try:
                    if caption_text and "<caption" not in html.lower():
                        m = re.search(r"<table[^>]*>", html, flags=re.IGNORECASE)
                        if m:
                            insert_at = m.end()
                            html = html[:insert_at] + f"<caption>{caption_text}</caption>" + html[insert_at:]
                except Exception:
                    logging.exception("Failed to insert caption into VLM HTML")

                # Use cropout for image and attach validated HTML content (VLM path)
                res.append((cropout(bxs, "table", poss), html))
                positions.append(poss)
                continue

            # Standard (deepdoc) table processing path remains unchanged
            res.append((cropout(bxs, "table", poss), self.tbl_det.construct_table(bxs, html=return_html, is_english=self.is_english)))
            positions.append(poss)

        if separate_tables_figures:
            assert len(positions) + len(figure_positions) == len(res) + len(figure_results)
            if need_position:
                return list(zip(res, positions)), list(zip(figure_results, figure_positions))
            else:
                return res, figure_results
        else:
            assert len(positions) == len(res)
            if need_position:
                return list(zip(res, positions))
            else:
                return res

    def proj_match(self, line):
        if len(line) <= 2:
            return
        if re.match(r"[0-9 ().,%%+/-]+$", line):
            return False
        for p, j in [
            (r"第[零一二三四五六七八九十百]+章", 1),
            (r"第[零一二三四五六七八九十百]+[条节]", 2),
            (r"[零一二三四五六七八九十百]+[、 　]", 3),
            (r"[\(（][零一二三四五六七八九十百]+[）\)]", 4),
            (r"[0-9]+(、|\.[　 ]|\.[^0-9])", 5),
            (r"[0-9]+\.[0-9]+(、|[. 　]|[^0-9])", 6),
            (r"[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 7),
            (r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 8),
            (r".{,48}[：:?？]$", 9),
            (r"[0-9]+）", 10),
            (r"[\(（][0-9]+[）\)]", 11),
            (r"[零一二三四五六七八九十百]+是", 12),
            (r"[⚫•➢✓]", 12),
        ]:
            if re.match(p, line):
                return j
        return

    def _line_tag(self, bx, ZM):
        pn = [bx["page_number"]]
        top = bx["top"] - self.page_cum_height[pn[0] - 1]
        bott = bx["bottom"] - self.page_cum_height[pn[0] - 1]
        page_images_cnt = len(self.page_images)
        if pn[-1] - 1 >= page_images_cnt:
            return ""
        while bott * ZM > self.page_images[pn[-1] - 1].size[1]:
            bott -= self.page_images[pn[-1] - 1].size[1] / ZM
            pn.append(pn[-1] + 1)
            if pn[-1] - 1 >= page_images_cnt:
                return ""

        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##".format("-".join([str(p) for p in pn]), bx["x0"], bx["x1"], top, bott)

    def __filterout_scraps(self, boxes, ZM):
        def width(b):
            return b["x1"] - b["x0"]

        def height(b):
            return b["bottom"] - b["top"]

        def usefull(b):
            if b.get("layout_type"):
                return True
            if width(b) > self.page_images[b["page_number"] - 1].size[0] / ZM / 3:
                return True
            if b["bottom"] - b["top"] > self.mean_height[b["page_number"] - 1]:
                return True
            return False

        res = []
        while boxes:
            lines = []
            widths = []
            pw = self.page_images[boxes[0]["page_number"] - 1].size[0] / ZM
            mh = self.mean_height[boxes[0]["page_number"] - 1]
            mj = self.proj_match(boxes[0]["text"]) or boxes[0].get("layout_type", "") == "title"

            def dfs(line, st):
                nonlocal mh, pw, lines, widths
                lines.append(line)
                widths.append(width(line))
                mmj = self.proj_match(line["text"]) or line.get("layout_type", "") == "title"
                for i in range(st + 1, min(st + 20, len(boxes))):
                    if (boxes[i]["page_number"] - line["page_number"]) > 0:
                        break
                    if not mmj and self._y_dis(line, boxes[i]) >= 3 * mh and height(line) < 1.5 * mh:
                        break

                    if not usefull(boxes[i]):
                        continue
                    if mmj or (self._x_dis(boxes[i], line) < pw / 10):
                        # and abs(width(boxes[i])-width_mean)/max(width(boxes[i]),width_mean)<0.5):
                        # concat following
                        dfs(boxes[i], i)
                        boxes.pop(i)
                        break

            try:
                if usefull(boxes[0]):
                    dfs(boxes[0], 0)
                else:
                    logging.debug("WASTE: " + boxes[0]["text"])
            except Exception:
                pass
            boxes.pop(0)
            mw = np.mean(widths)
            if mj or mw / pw >= 0.35 or mw > 200:
                res.append("\n".join([c["text"] + self._line_tag(c, ZM) for c in lines]))
            else:
                logging.debug("REMOVED: " + "<<".join([c["text"] for c in lines]))

        return "\n\n".join(res)

    @staticmethod
    def total_page_number(fnm, binary=None):
        try:
            with sys.modules[LOCK_KEY_pdfplumber]:
                pdf = pdfplumber.open(fnm) if not binary else pdfplumber.open(BytesIO(binary))
            total_page = len(pdf.pages)
            pdf.close()
            return total_page
        except Exception:
            logging.exception("total_page_number")

    def __images__(self, fnm, zoomin=3, page_from=0, page_to=299, callback=None):
        self.lefted_chars = []
        self.mean_height = []
        self.mean_width = []
        self.boxes = []
        self.garbages = {}
        self.page_cum_height = [0]
        self.page_layout = []
        self.page_from = page_from
        start = timer()
        try:
            with sys.modules[LOCK_KEY_pdfplumber]:
                with pdfplumber.open(fnm) if isinstance(fnm, str) else pdfplumber.open(BytesIO(fnm)) as pdf:
                    self.pdf = pdf
                    self.page_images = [p.to_image(resolution=72 * zoomin, antialias=True).annotated for i, p in enumerate(self.pdf.pages[page_from:page_to])]

                    try:
                        self.page_chars = [[c for c in page.dedupe_chars().chars if self._has_color(c)] for page in self.pdf.pages[page_from:page_to]]
                    except Exception as e:
                        logging.warning(f"Failed to extract characters for pages {page_from}-{page_to}: {str(e)}")
                        self.page_chars = [[] for _ in range(page_to - page_from)]  # If failed to extract, using empty list instead.

                    self.total_page = len(self.pdf.pages)

        except Exception:
            logging.exception("RAGFlowPdfParser __images__")
        logging.info(f"__images__ dedupe_chars cost {timer() - start}s")

        self.outlines = []
        try:
            with pdf2_read(fnm if isinstance(fnm, str) else BytesIO(fnm)) as pdf:
                self.pdf = pdf

                outlines = self.pdf.outline

                def dfs(arr, depth):
                    for a in arr:
                        if isinstance(a, dict):
                            self.outlines.append((a["/Title"], depth))
                            continue
                        dfs(a, depth + 1)

                dfs(outlines, 0)

        except Exception as e:
            logging.warning(f"Outlines exception: {e}")

        if not self.outlines:
            logging.warning("Miss outlines")

        logging.debug("Images converted.")
        self.is_english = [
            re.search(r"[a-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}", "".join(random.choices([c["text"] for c in self.page_chars[i]], k=min(100, len(self.page_chars[i])))))
            for i in range(len(self.page_chars))
        ]
        if sum([1 if e else 0 for e in self.is_english]) > len(self.page_images) / 2:
            self.is_english = True
        else:
            self.is_english = False

        async def __img_ocr(i, id, img, chars, limiter):
            j = 0
            while j + 1 < len(chars):
                if (
                    chars[j]["text"]
                    and chars[j + 1]["text"]
                    and re.match(r"[0-9a-zA-Z,.:;!%]+", chars[j]["text"] + chars[j + 1]["text"])
                    and chars[j + 1]["x0"] - chars[j]["x1"] >= min(chars[j + 1]["width"], chars[j]["width"]) / 2
                ):
                    chars[j]["text"] += " "
                j += 1

            if limiter:
                async with limiter:
                    await trio.to_thread.run_sync(lambda: self.__ocr(i + 1, img, chars, zoomin, id))
            else:
                self.__ocr(i + 1, img, chars, zoomin, id)

            if callback and i % 6 == 5:
                callback((i + 1) * 0.6 / len(self.page_images))

        async def __img_ocr_launcher():
            def __ocr_preprocess():
                chars = self.page_chars[i] if not self.is_english else []
                self.mean_height.append(np.median(sorted([c["height"] for c in chars])) if chars else 0)
                self.mean_width.append(np.median(sorted([c["width"] for c in chars])) if chars else 8)
                self.page_cum_height.append(img.size[1] / zoomin)
                return chars

            if self.parallel_limiter:
                async with trio.open_nursery() as nursery:
                    for i, img in enumerate(self.page_images):
                        chars = __ocr_preprocess()

                        nursery.start_soon(__img_ocr, i, i % PARALLEL_DEVICES, img, chars, self.parallel_limiter[i % PARALLEL_DEVICES])
                        await trio.sleep(0.1)
            else:
                for i, img in enumerate(self.page_images):
                    chars = __ocr_preprocess()
                    await __img_ocr(i, 0, img, chars, None)

        start = timer()

        trio.run(__img_ocr_launcher)

        logging.info(f"__images__ {len(self.page_images)} pages cost {timer() - start}s")

        if not self.is_english and not any([c for c in self.page_chars]) and self.boxes:
            bxes = [b for bxs in self.boxes for b in bxs]
            self.is_english = re.search(r"[\na-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}", "".join([b["text"] for b in random.choices(bxes, k=min(30, len(bxes)))]))

        logging.debug(f"Is it English: {self.is_english}")

        self.page_cum_height = np.cumsum(self.page_cum_height)
        assert len(self.page_cum_height) == len(self.page_images) + 1
        if len(self.boxes) == 0 and zoomin < 9:
            self.__images__(fnm, zoomin * 3, page_from, page_to, callback)

    def __call__(self, fnm, need_image=True, zoomin=3, return_html=False):
        self.__images__(fnm, zoomin)
        self._layouts_rec(zoomin)
        self._table_transformer_job(zoomin)
        self._text_merge()
        self._concat_downward()
        self._filter_forpages()
        tbls = self._extract_table_figure(need_image, zoomin, return_html, False)
        return self.__filterout_scraps(deepcopy(self.boxes), zoomin), tbls

    def parse_into_bboxes(self, fnm, callback=None, zoomin=3):
        start = timer()
        self.__images__(fnm, zoomin, callback=callback)
        if callback:
            callback(0.40, "OCR finished ({:.2f}s)".format(timer() - start))

        start = timer()
        self._layouts_rec(zoomin)
        if callback:
            callback(0.63, "Layout analysis ({:.2f}s)".format(timer() - start))

        start = timer()
        self._table_transformer_job(zoomin)
        if callback:
            callback(0.83, "Table analysis ({:.2f}s)".format(timer() - start))

        start = timer()
        self._text_merge()
        self._concat_downward()
        self._naive_vertical_merge(zoomin)
        if callback:
            callback(0.92, "Text merged ({:.2f}s)".format(timer() - start))

        start = timer()
        tbls, figs = self._extract_table_figure(True, zoomin, True, True, True)

        def insert_table_figures(tbls_or_figs, layout_type):
            def min_rectangle_distance(rect1, rect2):
                pn1, left1, right1, top1, bottom1 = rect1
                pn2, left2, right2, top2, bottom2 = rect2
                if right1 >= left2 and right2 >= left1 and bottom1 >= top2 and bottom2 >= top1:
                    return 0
                if right1 < left2:
                    dx = left2 - right1
                elif right2 < left1:
                    dx = left1 - right2
                else:
                    dx = 0
                if bottom1 < top2:
                    dy = top2 - bottom1
                elif bottom2 < top1:
                    dy = top1 - bottom2
                else:
                    dy = 0
                return math.sqrt(dx * dx + dy * dy)  # + (pn2-pn1)*10000

            for (img, txt), poss in tbls_or_figs:
                bboxes = [(i, (b["page_number"], b["x0"], b["x1"], b["top"], b["bottom"])) for i, b in enumerate(self.boxes)]
                dists = [
                    (min_rectangle_distance((pn, left, right, top + self.page_cum_height[pn], bott + self.page_cum_height[pn]), rect), i) for i, rect in bboxes for pn, left, right, top, bott in poss
                ]
                min_i = np.argmin(dists, axis=0)[0]
                min_i, rect = bboxes[dists[min_i][-1]]
                if isinstance(txt, list):
                    txt = "\n".join(txt)
                pn, left, right, top, bott = poss[0]
                if self.boxes[min_i]["bottom"] < top + self.page_cum_height[pn]:
                    min_i += 1
                self.boxes.insert(
                    min_i,
                    {
                        "page_number": pn + 1,
                        "x0": left,
                        "x1": right,
                        "top": top + self.page_cum_height[pn],
                        "bottom": bott + self.page_cum_height[pn],
                        "layout_type": layout_type,
                        "text": txt,
                        "image": img,
                        "positions": [[pn + 1, int(left), int(right), int(top), int(bott)]],
                    },
                )

        for b in self.boxes:
            b["position_tag"] = self._line_tag(b, zoomin)
            b["image"] = self.crop(b["position_tag"], zoomin)
            b["positions"] = [[pos[0][-1] + 1, *pos[1:]] for pos in RAGFlowPdfParser.extract_positions(b["position_tag"])]

        insert_table_figures(tbls, "table")
        insert_table_figures(figs, "figure")
        if callback:
            callback(1, "Structured ({:.2f}s)".format(timer() - start))
        return deepcopy(self.boxes)

    @staticmethod
    def remove_tag(txt):
        return re.sub(r"@@[\t0-9.-]+?##", "", txt)

    @staticmethod
    def extract_positions(txt):
        poss = []
        for tag in re.findall(r"@@[0-9-]+\t[0-9.\t]+##", txt):
            pn, left, right, top, bottom = tag.strip("#").strip("@").split("\t")
            left, right, top, bottom = float(left), float(right), float(top), float(bottom)
            poss.append(([int(p) - 1 for p in pn.split("-")], left, right, top, bottom))
        return poss

    def crop(self, text, ZM=3, need_position=False):
        imgs = []
        poss = self.extract_positions(text)
        if not poss:
            if need_position:
                return None, None
            return

        max_width = max(np.max([right - left for (_, left, right, _, _) in poss]), 6)
        GAP = 6
        pos = poss[0]
        poss.insert(0, ([pos[0][0]], pos[1], pos[2], max(0, pos[3] - 120), max(pos[3] - GAP, 0)))
        pos = poss[-1]
        poss.append(([pos[0][-1]], pos[1], pos[2], min(self.page_images[pos[0][-1]].size[1] / ZM, pos[4] + GAP), min(self.page_images[pos[0][-1]].size[1] / ZM, pos[4] + 120)))

        positions = []
        for ii, (pns, left, right, top, bottom) in enumerate(poss):
            right = left + max_width
            bottom *= ZM
            for pn in pns[1:]:
                bottom += self.page_images[pn - 1].size[1]
            imgs.append(self.page_images[pns[0]].crop((left * ZM, top * ZM, right * ZM, min(bottom, self.page_images[pns[0]].size[1]))))
            if 0 < ii < len(poss) - 1:
                positions.append((pns[0] + self.page_from, left, right, top, min(bottom, self.page_images[pns[0]].size[1]) / ZM))
            bottom -= self.page_images[pns[0]].size[1]
            for pn in pns[1:]:
                imgs.append(self.page_images[pn].crop((left * ZM, 0, right * ZM, min(bottom, self.page_images[pn].size[1]))))
                if 0 < ii < len(poss) - 1:
                    positions.append((pn + self.page_from, left, right, 0, min(bottom, self.page_images[pn].size[1]) / ZM))
                bottom -= self.page_images[pn].size[1]

        if not imgs:
            if need_position:
                return None, None
            return
        height = 0
        for img in imgs:
            height += img.size[1] + GAP
        height = int(height)
        width = int(np.max([i.size[0] for i in imgs]))
        pic = Image.new("RGB", (width, height), (245, 245, 245))
        height = 0
        for ii, img in enumerate(imgs):
            if ii == 0 or ii + 1 == len(imgs):
                img = img.convert("RGBA")
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                overlay.putalpha(128)
                img = Image.alpha_composite(img, overlay).convert("RGB")
            pic.paste(img, (0, int(height)))
            height += img.size[1] + GAP

        if need_position:
            return pic, positions
        return pic

    def get_position(self, bx, ZM):
        poss = []
        pn = bx["page_number"]
        top = bx["top"] - self.page_cum_height[pn - 1]
        bott = bx["bottom"] - self.page_cum_height[pn - 1]
        poss.append((pn, bx["x0"], bx["x1"], top, min(bott, self.page_images[pn - 1].size[1] / ZM)))
        while bott * ZM > self.page_images[pn - 1].size[1]:
            bott -= self.page_images[pn - 1].size[1] / ZM
            top = 0
            pn += 1
            poss.append((pn, bx["x0"], bx["x1"], top, min(bott, self.page_images[pn - 1].size[1] / ZM)))
        return poss
 
 
    def _validate_html_table(self, html_output):
        """
        Validate and clean HTML table output from VLM.
        - Remove markdown code fences (```html ... ```)
        - Ensure a <table> wrapper exists (wrap if missing)
        - Log a warning if no <tr> tags are present (likely malformed)
        - Return cleaned HTML string
        """
        try:
            if not isinstance(html_output, str):
                return html_output
            s = html_output.strip()
            # Remove markdown code fences like ```html ... ``` or ``` ... ```
            s = re.sub(r"```(?:\s*html)?\s*\n?", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\n?```", "", s)
            s = s.strip()
            lower = s.lower()
            # If there's no table tag at all, wrap the content in <table>...</table>
            if "<table" not in lower:
                logging.warning("VLM returned table-like content without <table> tag; wrapping in <table>...</table>")
                s = "<table>\n" + s + "\n</table>"
                lower = s.lower()
            # If there are no <tr> tags, log a warning (rows missing)
            if "<tr" not in lower:
                logging.warning("Validated HTML table appears to have no <tr> entries; output may be malformed.")
            return s
        except Exception:
            logging.exception("_validate_html_table")
            return html_output
 
    def _validate_markdown_table(self, md_output):
        """
        Validate markdown table output.
        - Ensure pipe '|' separators are present
        - Log a warning if format looks invalid
        - Return cleaned markdown string
        """
        try:
            if not isinstance(md_output, str):
                return md_output
            s = md_output.strip()
            # Remove surrounding markdown fences if present
            s = re.sub(r"```(?:\s*markdown)?\s*\n?", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\n?```", "", s)
            s = s.strip()
            # Basic validation: presence of pipe separators and at least one header separator line (---)
            if "|" not in s:
                logging.warning("Validated markdown table does not contain '|' separators; format may be invalid for a table.")
            else:
                # Check for a header separator like | --- | --- |
                if not re.search(r"\|\s*-{3,}\s*\|", s):
                    # Not mandatory to have header separator in all markdown styles, but warn if missing
                    logging.warning("Markdown table missing a header separator row (e.g. | --- | --- |); ensure proper markdown table formatting.")
            return s
        except Exception:
            logging.exception("_validate_markdown_table")
            return md_output

    def _get_table_prompt(self, output_format="html"):
        """
        Load table-specific VLM prompt.

        Behavior:
        - If VLM_TABLE_PROMPT_PATH env var is set, attempt to load that file.
        - Otherwise default to rag/prompts/table_vlm_prompt.md relative to repo root.
        - If file missing or unreadable, return an inline fallback prompt.
        - Supports output_format "html" or "markdown".
        """
        try:
            prompt_file = os.getenv("VLM_TABLE_PROMPT_PATH")
            if not prompt_file:
                base = Path(__file__).resolve().parent.parent.parent
                prompt_file = base / "rag" / "prompts" / "table_vlm_prompt.md"
            else:
                prompt_file = Path(prompt_file)

            if prompt_file.exists():
                try:
                    return prompt_file.read_text(encoding="utf-8")
                except Exception:
                    logging.exception("Failed to read table prompt file; falling back to inline prompt")
        except Exception:
            logging.exception("_get_table_prompt")

        # Inline fallback prompts
        fmt = str(output_format).lower()
        if fmt == "markdown" or fmt == "md":
            return (
                "Extract this table as markdown. Use '|' for columns and a header separator row (e.g. | --- | --- |). "
                "Preserve headers and data exactly. Output only the markdown table, without fences or any additional text."
            )
        else:
            return (
                "Extract this table as HTML. Use <table>, <tr>, <th>, <td> tags. "
                "Use integer colspan and rowspan attributes for merged cells. "
                "Include a <caption> if visible. Output ONLY the single <table>...</table> element, "
                "no surrounding HTML, markdown fences, or explanatory text."
            )
    
    def _vlm_table_parser(self, table_images, table_positions, vision_model=None):
        """
        Parse table images using a Vision-Language Model (VLM).
        
        Args:
            table_images: List of PIL images representing tables
            table_positions: List of positions for the tables
            vision_model: Vision model to use for parsing (LLMBundle or compatible)
            
        Returns:
            List of parsed table outputs (HTML/Markdown strings or None for fallback)
        """
        start_time = timer()
        logging.info(f"_vlm_table_parser: Starting to process {len(table_images)} table images")
        logging.debug(f"_vlm_table_parser: vision_model type={type(vision_model)}, available={vision_model is not None}")
        
        # Log environment settings that affect VLM behavior
        resize_factor = int(os.getenv("VLM_RESIZE_FACTOR", "32"))
        timeout_env = os.getenv("VLM_TABLE_TIMEOUT_SEC")
        output_format = os.getenv("VLM_TABLE_OUTPUT_FORMAT", "html").lower()
        fallback_enabled = os.getenv("VLM_TABLE_FALLBACK_ENABLED", "true").lower() == "true"
        
        logging.info(f"_vlm_table_parser: Configuration - resize_factor={resize_factor}, timeout={timeout_env}, output_format={output_format}, fallback_enabled={fallback_enabled}")
        
        results = []
        MAX_JPG_BYTES = 5 * 1024 * 1024  # 5 MB safety threshold
    
        for idx, img in enumerate(table_images):
            table_start_time = timer()
            try:
                if img is None:
                    raise ValueError("Received None image for table index {}".format(idx))
    
                # Ensure RGB
                pil = img.convert("RGB")
    
                # Compute resized dims (smart_resize expects height, width)
                width, height = pil.size
                target_h, target_w = smart_resize(height, width, factor=resize_factor, target_max_dimension=1024)
    
                # Resize with LANCZOS
                pil = pil.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
    
                # Convert to JPEG bytes
                buf = BytesIO()
                pil.save(buf, format="JPEG", quality=90, optimize=True)
                jpg_bytes = buf.getvalue()
    
                # If too large, attempt progressive reductions (quality -> shrink)
                if len(jpg_bytes) > MAX_JPG_BYTES:
                    logging.warning(f"_vlm_table_parser: table {idx} jpeg {len(jpg_bytes)} bytes > {MAX_JPG_BYTES}, trying quality=60")
                    buf = BytesIO()
                    pil.save(buf, format="JPEG", quality=60, optimize=True)
                    jpg_bytes = buf.getvalue()
    
                if len(jpg_bytes) > MAX_JPG_BYTES:
                    logging.warning(f"_vlm_table_parser: table {idx} still > {MAX_JPG_BYTES}, resizing down by 0.5")
                    new_size = (max(100, int(pil.size[0] * 0.5)), max(10, int(pil.size[1] * 0.5)))
                    pil_small = pil.resize(new_size, resample=Image.Resampling.LANCZOS)
                    buf = BytesIO()
                    pil_small.save(buf, format="JPEG", quality=60, optimize=True)
                    jpg_bytes = buf.getvalue()
    
                # Build prompt
                prompt = self._get_table_prompt(output_format)
                logging.debug(f"_vlm_table_parser: table {idx} prompt length: {len(prompt)} chars")
    
                # Call the vision LLM
                try:
                    if timeout is not None:
                        out = vision_llm_chunk(binary=jpg_bytes, vision_model=vision_model, prompt=prompt, timeout=timeout)
                    else:
                        out = vision_llm_chunk(binary=jpg_bytes, vision_model=vision_model, prompt=prompt)
                except TypeError:
                    # If the function does not accept timeout kwarg, retry without it
                    out = vision_llm_chunk(binary=jpg_bytes, vision_model=vision_model, prompt=prompt)
    
                # Unwrap tuple responses (some wrappers return (text, meta))
                if isinstance(out, tuple) and len(out) > 0:
                    out = out[0]
    
                # Normalize type
                if out is None:
                    raise RuntimeError("VLM returned None")
                if not isinstance(out, str):
                    try:
                        out = str(out)
                    except Exception:
                        raise RuntimeError("VLM returned non-string output")
    
                # Validate according to chosen format
                if output_format in ("html", "htm"):
                    validated = self._validate_html_table(out)
                else:
                    validated = self._validate_markdown_table(out)
                    
                # Calculate duration for this table
                table_duration = timer() - table_start_time
                logging.info(f"_vlm_table_parser: table {idx} processed in {table_duration:.2f}s")
    
                results.append(validated)
            except Exception as e:
                table_duration = timer() - table_start_time
                logging.exception(f"_vlm_table_parser: failed parsing table {idx} after {table_duration:.2f}s - Error: {str(e)}")
                if fallback_enabled:
                    # Signal caller to fallback by returning None for this index
                    results.append(None)
                else:
                    # Provide a minimal error table matching requested format
                    if output_format in ("html", "htm"):
                        results.append("<table><tr><td>Table parsing failed</td></tr></table>")
                    else:
                        results.append("| Table parsing failed |")
    
        total_duration = timer() - start_time
        logging.info(f"_vlm_table_parser: completed processing {len(table_images)} tables in {total_duration:.2f}s")
        return results
 
class PlainParser:
    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        self.outlines = []
        lines = []
        try:
            self.pdf = pdf2_read(filename if isinstance(filename, str) else BytesIO(filename))
            for page in self.pdf.pages[from_page:to_page]:
                lines.extend([t for t in page.extract_text().split("\n")])

            outlines = self.pdf.outline

            def dfs(arr, depth):
                for a in arr:
                    if isinstance(a, dict):
                        self.outlines.append((a["/Title"], depth))
                        continue
                    dfs(a, depth + 1)

            dfs(outlines, 0)
        except Exception:
            logging.exception("Outlines exception")
        if not self.outlines:
            logging.warning("Miss outlines")

        return [(line, "") for line in lines], []

    def crop(self, ck, need_position):
        raise NotImplementedError

    @staticmethod
    def remove_tag(txt):
        raise NotImplementedError


class VisionParser(RAGFlowPdfParser):
    def __init__(self, vision_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vision_model = vision_model

    def __images__(self, fnm, zoomin=3, page_from=0, page_to=299, callback=None):
        try:
            with sys.modules[LOCK_KEY_pdfplumber]:
                self.pdf = pdfplumber.open(fnm) if isinstance(fnm, str) else pdfplumber.open(BytesIO(fnm))
                # Extract images at high DPI (600) to ensure high quality for VLM processing
                high_dpi = 600
                self.page_images = [p.to_image(resolution=high_dpi).annotated for i, p in enumerate(self.pdf.pages[page_from:page_to])]
                self.total_page = len(self.pdf.pages)
        except Exception:
            self.page_images = None
            self.total_page = 0
            logging.exception("VisionParser __images__")

    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        # Extract prompt_text from kwargs
        prompt_text = kwargs.get("prompt_text", None)
        callback = kwargs.get("callback", lambda prog, msg: None)
        zoomin = kwargs.get("zoomin", 3)

        # Validate zoomin
        if not isinstance(zoomin, (int, float)) or zoomin <= 0:
            logging.error(f"Invalid zoomin value: {zoomin}, using default 3")
            zoomin = 3

        # Validate page range types and values
        try:
            from_page = int(from_page)
        except Exception:
            logging.warning(f"Invalid from_page type: {from_page}, using 0")
            from_page = 0
        try:
            to_page = int(to_page)
        except Exception:
            logging.warning(f"Invalid to_page type: {to_page}, using 100000")
            to_page = 100000

        if from_page < 0:
            logging.warning(f"Invalid from_page: {from_page}, using 0")
            from_page = 0
        if to_page < from_page:
            logging.error(f"Invalid page range: from={from_page}, to={to_page}")
            return [], []

        # Validate vision model presence
        if not getattr(self, "vision_model", None):
            logging.error("VisionParser: vision_model is not set or not configured")
            return [], []

        # Extract images (may set self.page_images)
        self.__images__(fnm=filename, zoomin=zoomin, page_from=from_page, page_to=to_page, callback=callback)

        # Check if images were extracted
        if not getattr(self, "page_images", None):
            logging.warning(f"No images extracted from {filename}")
            return [], []

        total_pdf_pages = self.total_page

        start_page = max(0, from_page)
        end_page = min(to_page, total_pdf_pages)

        # Summary info
        try:
            img_cnt = len(self.page_images) if self.page_images else 0
        except Exception:
            img_cnt = 0
        logging.info(f"VisionParser: Processing {img_cnt} pages (from={from_page}, to={to_page}, total_pdf_pages={total_pdf_pages})")

        all_docs = []

        for idx, img_pil in enumerate(self.page_images or []):
            pdf_page_num = idx  # 0-based
            if pdf_page_num < start_page or pdf_page_num >= end_page:
                continue

            # Preserve original page image size for metadata
            orig_width, orig_height = img_pil.size
            logging.debug(f"VisionParser: Page {idx+1}/{img_cnt}: Original size {orig_width}x{orig_height}")

            try:
                # Convert to RGB
                img = img_pil.convert("RGB")

                # Get resize factor from environment variable or default to 32
                resize_factor = int(os.getenv("VLM_RESIZE_FACTOR", "32"))
                
                # Apply smart_resize to ensure dimensions are multiples of the factor
                # and maintain proper aspect ratio with max 1024 on the long dimension
                width, height = img.size
                target_height, target_width = smart_resize(
                    height, width,
                    factor=resize_factor,
                    target_max_dimension=1024  # Balanced resolution for general use
                )
                
                # Resize the image to the calculated dimensions
                img = img.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)
                logging.debug(f"VisionParser: Page {idx+1}: Resized from {width}x{height} to {target_width}x{target_height} (factor={resize_factor})")

                # Convert to JPEG bytes
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=90, optimize=True)
                jpg_bytes = buf.getvalue()
                logging.debug(f"VisionParser: Page {idx+1}: JPEG bytes: {len(jpg_bytes)} bytes")

                # If JPEG bytes are very large, attempt additional compression rather than failing
                MAX_JPG_BYTES = 5 * 1024 * 1024  # 5MB
                if len(jpg_bytes) > MAX_JPG_BYTES:
                    logging.warning(f"VisionParser: Page {idx+1}: JPEG size {len(jpg_bytes)} exceeds {MAX_JPG_BYTES} bytes, attempting extra compression")
                    # try lower quality compression
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=60, optimize=True)
                    jpg_bytes = buf.getvalue()
                    logging.debug(f"VisionParser: Page {idx+1}: After compress quality=60 size={len(jpg_bytes)}")
                    # if still too large, resize down further
                    if len(jpg_bytes) > MAX_JPG_BYTES:
                        scale = 0.5
                        new_size = (max(100, int(img.size[0] * scale)), max(100, int(img.size[1] * scale)))
                        img_small = img.resize(new_size, resample=Image.Resampling.LANCZOS)
                        buf = io.BytesIO()
                        img_small.save(buf, format="JPEG", quality=60, optimize=True)
                        jpg_bytes = buf.getvalue()
                        logging.debug(f"VisionParser: Page {idx+1}: After shrink size={len(jpg_bytes)}")

                # Build prompt: use provided prompt_text (with page interpolation) or default generator
                if prompt_text is not None:
                    prompt = prompt_text.replace("{{ page }}", str(pdf_page_num + 1)) if "{{ page }}" in prompt_text else prompt_text
                else:
                    prompt = vision_llm_describe_prompt(page=pdf_page_num + 1)

                # Log prompt preview/length (trim to avoid huge logs)
                try:
                    preview = (prompt[:200] + "...") if len(str(prompt)) > 200 else prompt
                    logging.debug(f"VisionParser: Page {idx+1}: Prompt length: {len(str(prompt))} chars, preview: {preview}")
                except Exception:
                    logging.debug(f"VisionParser: Page {idx+1}: Prompt prepared (unable to render preview)")

                # Call VLM with page-level try/except to avoid whole-document failure
                try:
                    text = picture_vision_llm_chunk(
                        binary=jpg_bytes,
                        vision_model=self.vision_model,
                        prompt=prompt,
                        callback=callback,
                    )
                except Exception as e:
                    logging.error(f"Page {pdf_page_num + 1}: VLM call failed: {e}")
                    text = f"[Page {pdf_page_num + 1}: Processing error - {str(e)[:100]}]"
                # Progress/info per page
                logging.info(f"VisionParser: Page {idx+1}/{img_cnt} processed by VLM")

                if kwargs.get("callback"):
                    try:
                        kwargs["callback"](idx * 1.0 / img_cnt if img_cnt else 0.0, f"Processed: {idx + 1}/{img_cnt}")
                    except Exception:
                        pass

                # Normalize and robustly check VLM text
                if isinstance(text, tuple) and len(text) >= 1:
                    # Some models may return (text, tokens)
                    text = text[0]
                if text is None:
                    logging.warning(f"Page {pdf_page_num + 1}: VLM returned None or no text")
                    text = f"[Page {pdf_page_num + 1}: No content detected by VLM]"
                if not isinstance(text, str):
                    logging.warning(f"Page {pdf_page_num + 1}: VLM returned non-string ({type(text)}), coercing to str")
                    try:
                        text = str(text)
                    except Exception:
                        text = f"[Page {pdf_page_num + 1}: Non-string VLM output]"

                # Cleanup and basic heuristics
                cleaned = text.strip()
                # Handle very short responses
                if not cleaned or len(cleaned) < 10:
                    logging.warning(f"Page {pdf_page_num + 1}: Empty or very short VLM response ('{cleaned}')")
                    cleaned = f"[Page {pdf_page_num + 1}: No content detected by VLM]"

                # Detect possible gibberish (low vocabulary ratio)
                words = re.findall(r"\w+", cleaned)
                if len(words) > 20:
                    unique_words = set(words)
                    vocab_ratio = len(unique_words) / len(words) if len(words) else 0.0
                    if vocab_ratio < 0.3:
                        logging.warning(f"Page {pdf_page_num + 1}: Possible gibberish detected (vocab ratio: {vocab_ratio:.2f})")

                # Detect repeated patterns that might indicate model confusion
                if len(cleaned) > 100:
                    for pattern_len in (50, 100):
                        if len(cleaned) >= pattern_len * 3:
                            pattern = cleaned[:pattern_len]
                            if cleaned.count(pattern) >= 3:
                                logging.warning(f"Page {pdf_page_num + 1}: Detected repeated pattern, possible model error")
                                break

                # Warn on excessively long responses
                if len(cleaned) > 50000:
                    logging.warning(f"Page {pdf_page_num + 1}: Very long VLM response: {len(cleaned)} chars, consider truncation")

                width, height = orig_width, orig_height
                all_docs.append((
                    cleaned,
                    f"@@{pdf_page_num + 1}\t{0.0:.1f}\t{width / zoomin:.1f}\t{0.0:.1f}\t{height / zoomin:.1f}##"
                ))
            except Exception as e:
                logging.error(f"Page {pdf_page_num + 1}: Processing failed: {e}")
                # Add fallback entry instead of crashing whole document
                all_docs.append((
                    f"[Page {pdf_page_num + 1}: Processing error - {str(e)[:100]}]",
                    f"@@{pdf_page_num + 1}\t{0.0:.1f}\t{orig_width / zoomin:.1f}\t{0.0:.1f}\t{orig_height / zoomin:.1f}##"
                ))
                # continue to next page
                continue
        return all_docs, []


if __name__ == "__main__":
    pass
