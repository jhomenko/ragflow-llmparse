Primary feature addition
1) DONE In pdf_parser.py, which I believe contains the resizing and quality logic for the image sent to the vlm, I believe we may not be meeting the image input requirements for the vlm’s. For Qwen3-vl based models each dimension must be a multiple of 32 and for qwen2.5-vl based models it must be a multiple of 28. I’ve downloaded a reference utility module for qwen3 that has the “smart_resize” function, example code of this working from them is below, but I’ve also found some examples that do the following as a best practice. Extract the jpeg from the original pdf at a high dpi like 600, then resize to a maximum size of 1024 (tokens I believe) on the long dimension of the image while maintaining the aspect ratio of the image, rounded to the nearest 32 pixels:
    from qwen_vl_utils import smart_resize
    # os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here'
    min_pixels = 512*32*32
    max_pixels = 4608*32*32
    width, height = image.size
    input_height, input_width = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels, factor=32)
    output = inference_with_api(img_name, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
 

Moderate priority
Maybe complex. 2 parts, questions then options for features. I am not sure if I am asking this clearly enough so let me know your interpretation of what i'm asking, as well as the level of complexity and your preferred choice (or both) to implement.
Question: Looking at parser.py, for class Parser(ProcessBase), in the original module this class returned lines, bboxes, and had settings for output format json vs markdown. In our version of parser.py if i'm not mistaken since we essentially only ask our vlm to return markdown as the lines variable the class can't then parse the location of the parsed items in the final output right?
- looking at parser.py, is it still appropriate that it seems to try to find a valid zoomin value for the visionparser class or is this replaced by our smart image scaling (in the case of using vlm only, zoomin still required for other parsing methods)
Feature: 
Option 1
  Once we review the expected schema returned by Parser(ProcessBase) for _pdf, what would we need to change to align the response from the vlm to output the bboxes as well as lines? The prompt, obviously, but then also the response would need to be handled to extract those items, so we would want to actually request json from the model I think?
Option 2 (more complicated)
Would it be complex to modify pdf_parser.py to have selectable (by environment variable) hybrid functionality for table parsing, where instead of routing tables to table_structure_recognizer.py for table parsing, the area containing the table is converted to a base 64 image using existing code, and then sent to the llm module we’ve made for table parsing? The returned result can then be passed back into the pdf parser in the format that it is expecting. I think the method is defined in pdf_parser.py around 798-895 but i am not positive. Another upside of this functionality would be that it seems like table that break across pages are automatically concatendated by the default parser. I believe the existing code path is to identify the table  locations with pdf_parser as I mentioned, then the coordinates are sent to figure_parser.py. In this case, we would want to select the deepdoc parsing method, but then have this alternate function and the model name selected by environment variables. If we are able to implement this we will get the benefits of the speed and structured approach of the deepdoc modules while getting enhanced table parsing from the vlm which was the weak point before. The challenge will be ensuring the response from the vlm is the exact schema expected so the rest of the parsing flow proceeds without error.



Lower priority
- Can we add environment variables to allow us to change the max_tokens and temperature parameters of the vlm?

Question:


- It looks like there is a hard coded model in working_vlm_module.py, is that actually the case or just a placeholder default that’s replaced by the model selected in the ui?
- We have my actual ip address for the vlm server hardcoded as a default as well as setting as an env. We should only keep the env reference because it doesn’t make sense to hard code my own ip address in the repo.
- Will batching work automatically if I increase the -np in my server? Concurrent page processing will really speed things up.
- Configurable chunking strategies mentioned in implementation guide? is that in rag_tokenizer.py? I don't see where that would be can you explain this feature
- Can you explain this feature too? "VLM module adds multimodal pre-processing stage"
- Can you explain the parser configuration json portion? Because i believe these are all selected within the UI


- 