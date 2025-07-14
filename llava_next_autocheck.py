import argparse
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, LlavaNextProcessor, LlavaNextForConditionalGeneration
from llava.mm_utils import process_images
from utils.file_io import read_jsonlines, write_jsonlines

def yes_probability(logits, yes_ids, no_ids):
    probs = torch.softmax(logits, dim=-1)
    yes_prob = probs[yes_ids].sum().item()
    no_prob = probs[no_ids].sum().item()
    return yes_prob / (yes_prob + no_prob + 1e-8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--ds_name', type=str, required=True)
    parser.add_argument('--answer_file', type=str, required=True)
    parser.add_argument('--start_pos', type=int, default=0)
    parser.add_argument('--end_pos', type=int, default=-1)
    parser.add_argument('--is_yesno', action='store_true')
    args = parser.parse_args()

    length = None
    if args.end_pos != -1:
        length = args.end_pos - args.start_pos

    if length is not None:
        data = read_jsonlines(args.ds_name, args.start_pos, length)
    else:
        data = read_jsonlines(args.ds_name, args.start_pos)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    image_processor = CLIPImageProcessor.from_pretrained(args.checkpoint)
    processor = LlavaNextProcessor(tokenizer=tokenizer, image_processor=image_processor)
    
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.checkpoint,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
    model.eval()

    yes_ids = [processor.tokenizer.encode(' yes')[-1], processor.tokenizer.encode(' Yes')[-1]]
    no_ids = [processor.tokenizer.encode(' no')[-1], processor.tokenizer.encode(' No')[-1]]

    results = []
    for item in tqdm(data, desc='Autochecking with Vision'):
        image_path = item.get('metainfos', {}).get('image_path')
        if not image_path:
            print(f"Warning: image_path not found for question_id {item.get('question_id')}. Skipping.")
            continue
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image file not found at {image_path}. Skipping.")
            continue
        
        image_sizes = [image.size]

        sub_scores = []
        for sub in item.get('sub_sents', []):
            prompt_text = f"Based on the image, is the following statement true? Statement: \"{sub}\"\nPlease answer with only 'yes' or 'no'."
            inputs = processor(text=prompt_text, images=image, return_tensors='pt').to(model.device)

            with torch.no_grad():
                # This is the line with the correction.
                output = model.generate(**inputs, image_sizes=image_sizes, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
                
            logits = output.scores[0][0]
            score = yes_probability(logits, yes_ids, no_ids)
            sub_scores.append(score)
        
        item['scores'] = sub_scores
        item['score'] = sum(sub_scores)/len(sub_scores) if sub_scores else 0.0
        results.append(item)

    write_jsonlines(args.answer_file, results)

if __name__ == '__main__':
    main()
