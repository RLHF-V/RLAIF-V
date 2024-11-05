import os
import sys
import ssl
import json
import copy
import glob
import time
import pathlib
import random
import jsonlines

import nltk
import spacy
import argparse
import concurrent.futures

from concurrent.futures import ThreadPoolExecutor
from nltk.stem import *
from gpt4_grpc import Chat
from tqdm import tqdm


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nlp = spacy.load("en_core_web_trf")
lemma = nltk.wordnet.WordNetLemmatizer()


def parse_object_list(content):
    try:
        content = json.loads(content)
    except:
        if '["' in content:
            try:
                content = json.loads(content.strip().split('\n')[-1])
            except:
                raise ValueError('Content is not json interpretable')
        else:
            raise ValueError('Content is not json interpretable')
    return content


prompt_template = """You are an expert in image objects extraction according to a question answer pair. We asked an examiner to answer a question about a picture.

[Start of Question]

<image> {question}

[End of Question]

[Start of Examiner's Answer]

{answer}

[End of Examiner's Answer]


Assume that the answer is correct, please identify all visible objects that are directly shown in the image. Please following the instructions in below:

1. You should only mention objects that are explicitly mentioned in the examiner's answer.
2. You should only extract the object names without the attributes of the objects.
3. You should not include the properties of the object, like the color, material, etc. as part of the object name in your result.
4. Make your answer precise. Present the results in a JSON list format: [\"object_1\", ..., \"object_n\"].
5. You should return an empty JSON list () if no visible objects can be found.
"""

def preprocess_coh_results(caps):
    new_caps = []
    for cap in caps:
        cap_text = cap["caption"]
        if "The following is a response without hallucination." in cap_text:
            new_cap_text = cap_text.split("The following is a response without hallucination.")[-1].strip()
        elif "The following is a response with hallucination." in cap_text:
            new_cap_text = cap_text.split("The following is a response with hallucination.")[0].strip()
        elif "Generate a response without errors." in cap_text:
            new_cap_text = cap_text.split("Generate a response without errors.")[-1].strip()
        elif "Generate a response with errors." in cap_text:
            new_cap_text = cap_text.split("Generate a response with errors.")[0].strip()
        else:
            new_cap_text = cap_text
        cap['caption'] = new_cap_text
        new_caps.append(cap)

    return new_caps

def combine_coco_captions(annotation_path):

    if not os.path.exists('%s/captions_%s2014.json' %(annotation_path, 'val')):
        raise Exception("Please download MSCOCO caption annotations for val set")
    if not os.path.exists('%s/captions_%s2014.json' %(annotation_path, 'train')):
        raise Exception("Please download MSCOCO caption annotations for train set")

    val_caps = json.load(open('%s/captions_%s2014.json' %(annotation_path, 'val')))
    train_caps = json.load(open('%s/captions_%s2014.json' %(annotation_path, 'train')))
    all_caps = {'info': train_caps['info'],
                'licenses': train_caps['licenses'],
                'images': val_caps['images'] + train_caps['images'],
                'annotations': val_caps['annotations'] + train_caps['annotations']}

    return all_caps

def combine_coco_instances(annotation_path):

    if not os.path.exists('%s/instances_%s2014.json' %(annotation_path, 'val')):
        raise Exception("Please download MSCOCO instance annotations for val set")
    if not os.path.exists('%s/instances_%s2014.json' %(annotation_path, 'train')):
        raise Exception("Please download MSCOCO instance annotations for train set")

    val_instances = json.load(open('%s/instances_%s2014.json' %(annotation_path, 'val')))
    train_instances = json.load(open('%s/instances_%s2014.json' %(annotation_path, 'train')))
    all_instances = {'info': train_instances['info'],
                     'licenses': train_instances['licenses'],
                     'type': train_instances['licenses'],
                     'categories': train_instances['categories'],
                     'images': train_instances['images'] + val_instances['images'],
                     'annotations': val_instances['annotations'] + train_instances['annotations']}

    return all_instances

class CHAIR(object):

    def __init__(self, imids, coco_path, openai_apikey):

        self.imid_to_objects = {imid: [] for imid in imids}

        self.coco_path = coco_path

        self.chat_model = Chat(model="gpt-3.5-turbo-0613", timeout_sec=100, openai_apikey=openai_apikey)
        self.fail_limit=100


        #read in synonyms
        synonyms = open('./eval/data/synonyms_refine.txt').readlines()
        synonyms = [s.strip().split(', ') for s in synonyms]
        self.mscoco_objects = [] #mscoco objects and *all* synonyms
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            new_synonym = [s.strip() for s in synonym]
            self.mscoco_objects.extend(new_synonym)
            for s in new_synonym:
                self.inverse_synonym_dict[s] = new_synonym[0]

        coco_double_words = [word for word in self.inverse_synonym_dict.keys() if len(word.strip().split(' ')) >= 2]
        coco_double_words += ['home plate', 'train track']
        print("double word count:", len(coco_double_words))

        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
        vehicle_words = ['jet', 'train']

        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict['baby %s' %animal_word] = animal_word
            self.double_word_dict['adult %s' %animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' %vehicle_word] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'

    def _load_generated_captions_into_evaluator(self, cap_file, sample_num, org_dir=None):

        '''
        Meant to save time so imid_to_objects does not always need to be recomputed.
        '''
        #Read in captions
        self.caps, imids, self.metrics = load_generated_captions(cap_file, org_dir=org_dir)
        self.caps = list(self.caps)
        for index, cap in enumerate(self.caps):
            cap['index'] = index
        if sample_num != -1:
            self.caps = random.sample(self.caps, sample_num)
        print("cal cap num:", len(self.caps))

        assert imids == set(self.imid_to_objects.keys())

    def get_double_words_only(self, word_list):
        i = 0
        double_words = []
        idxs = []
        words = word_list
        while i < len(words):
           idxs.append(i)
           double_word = ' '.join(words[i:i+2])
           if double_word in self.double_word_dict:
               double_words.append(self.double_word_dict[double_word])
               i += 2
           else:
            #    double_words.append(words[i])
               i += 1
        words = double_words

        return words

    def caption_to_words(self, caption):

        '''
        Input: caption
        Output: MSCOCO words in the caption
        '''

        words = nltk.word_tokenize(caption.lower())
        words_2 = [lemma.lemmatize(w) for w in words]
        words = words_2

        #replace double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
           idxs.append(i)
           double_word = ' '.join(words[i:i+2])
           if double_word in self.double_word_dict:
               double_words.append(self.double_word_dict[double_word])
               i += 2
           else:
               double_words.append(words[i])
               i += 1
        words = double_words

        #toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
        if ('toilet' in words) & ('seat' in words): words = [word for word in words if word != 'seat']

        #get synonyms for all words in the caption
        idxs = [idxs[idx] for idx, word in enumerate(words) \
                if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append([word, self.inverse_synonym_dict[word]])
        #return all the MSCOCO objects in the caption
        return words, node_words, idxs, double_words

    def caption_objects_to_coco_objects(self, words):
        idxs = list(range(len(words)))
        if ('toilet' in words) & ('seat' in words): words = [word for word in words if word != 'seat']
        #get synonyms for all words in the caption
        idxs = [idxs[idx] for idx, word in enumerate(words) \
                if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append([word, self.inverse_synonym_dict[word]])

        #return all the MSCOCO objects in the caption
        return words, node_words, idxs

    def get_annotations_from_segments(self):
        '''
        Add objects taken from MSCOCO segmentation masks
        '''

        coco_segments = combine_coco_instances(self.coco_path)
        segment_annotations = coco_segments['annotations']

        id_to_name = {} #dict with id to synsets
        for cat in coco_segments['categories']:
            id_to_name[cat['id']] = cat['name']


        for i, annotation in enumerate(segment_annotations):
            sys.stdout.write("\rGetting annotations for %d/%d segmentation masks"
                              %(i, len(segment_annotations)))
            imid = annotation['image_id']
            if imid in self.imid_to_objects:
                node_word = self.inverse_synonym_dict[id_to_name[annotation['category_id']]]
                self.imid_to_objects[imid].append(node_word)
        print("\n")
        for imid in self.imid_to_objects:
            self.imid_to_objects[imid] = set(self.imid_to_objects[imid])

    def get_annotations_from_captions(self):
        '''
        Add objects taken from MSCOCO ground truth captions
        '''

        coco_caps = combine_coco_captions(self.coco_path)
        caption_annotations = coco_caps['annotations']

        for i, annotation in enumerate(caption_annotations):
            sys.stdout.write('\rGetting annotations for %d/%d ground truth captions'
                            %(i, len(coco_caps['annotations'])))
            imid = annotation['image_id']
            if imid in self.imid_to_objects:
                _, node_words, _, _ = self.caption_to_words(annotation['caption'])
                self.imid_to_objects[imid].update([item[-1] for item in node_words])
        print("\n")

        for imid in self.imid_to_objects:
            self.imid_to_objects[imid] = set(self.imid_to_objects[imid])

    def get_annotations(self):

        '''
        Get annotations from both segmentation and captions.  Need both annotation types for CHAIR metric.
        '''

        self.get_annotations_from_segments()
        self.get_annotations_from_captions()

    def get_gpt_resp(self, data_item):
        prompt = copy.deepcopy(prompt_template)
        prompt = prompt.replace("{question}", data_item["question"])
        prompt = prompt.replace("{answer}", data_item["caption"])

        messages = [
            {"role": "system", "content": prompt}
        ]

        fail_cnt = 0
        used_tokens = {"total": 0, "input": 0, "output": 0}
        while True:
            if len(data_item["caption"].strip().split()) <= 3:
                data_item["extract_objs"] = []
                print(f"**[Short Answer]**@{data_item['caption']}@", data_item["extract_objs"])
                return data_item, used_tokens, {"total": 0, "input": 0, "output": 0}

            if fail_cnt == self.fail_limit:
                data_item["extract_objs"] = f'-1\n<no_response>'
                print("**[Wrong Return]**", data_item["extract_objs"])
                return data_item, used_tokens, {"total": 0, "input": 0, "output": 0}

            resp = None
            try:
                resp = self.chat_model.chat_completion(messages=messages)
                print(resp["model"])

                # Logging consumption
                used_tokens["total"] += resp['usage']["total_tokens"]
                used_tokens["input"] += resp['usage']["prompt_tokens"]
                used_tokens["output"] += resp['usage']["completion_tokens"]

                # Parsing ChatGPT response
                content = resp["choices"][0]["message"]["content"]
                content = parse_object_list(content)

                # API Rest
                time.sleep(5)

                data_item["extract_objs"] = content
                success_tokens = {"total": resp['usage']['total_tokens'],
                                  "input": resp['usage']['prompt_tokens'],
                                  "output": resp['usage']['completion_tokens']}
                return data_item, used_tokens, success_tokens
            except Exception as e:
                fail_cnt += 1
                # print(f'{data_item["index"]} Fail for other reasons')
                # print("message:", messages)
                print("Exception:", e, 'resp is ', resp)

                time.sleep(10 + fail_cnt)

    def gpt_caption_processor(self, max_workers=64):
        data_list = self.caps
        new_data = []
        all_used_tokens = {"total": 0, "input": 0, "output": 0}
        all_success_tokens = {"total": 0, "input": 0, "output": 0}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            print("thread num:", len(data_list))

            futures = [
                executor.submit(self.get_gpt_resp, data_item)
                for data_item in data_list
            ]

            pb = tqdm(total=len(futures))

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                pb.update(1)

                try:
                    new_data_item, used_tokens, success_tokens = future.result() # type = List
                    all_used_tokens = {key: all_used_tokens[key] + used_tokens[key] for key in all_used_tokens.keys()}
                    all_success_tokens = {key: all_success_tokens[key] + success_tokens[key] for key in all_success_tokens.keys()}
                    new_data.append(new_data_item)

                except Exception as e:
                    print(f"@@@ Exception: {e}\n")
        print(f'Done loop, waiting resource finalization', flush=True)

        return new_data, all_used_tokens, all_success_tokens

    def postagging(self, doc):
        obj_list = []
        temp_token = ""

        for token in doc:
            if token.tag_ in ["NNP", "NNPS", "NN", "NNS"]:
                temp_token += f" {token.lemma_}"
            else:
                if temp_token != "":
                    obj_list.append(temp_token.strip())
                    temp_token = ""
        if temp_token != "":
            obj_list.append(temp_token.strip())
            temp_token = ""
        return obj_list

    def get_pred_objs_match(self, caps):
        new_caps = []
        for item in caps:
            caps_gpt_objs = item["extract_objs"]
            assert caps_gpt_objs != f'-1\n<no_response>'
            refined_objs = []
            for text in caps_gpt_objs:
                text = f"a {text}"
                doc = nlp(text)
                single_tokens = [token.lemma_ for token in doc]
                double_words_objs = self.get_double_words_only(single_tokens)

                if double_words_objs != []:
                    refined_objs += double_words_objs
                    continue

                postagging_objs = self.postagging(doc)
                refined_objs += postagging_objs

            new_item = copy.deepcopy(item)

            # only append unique word in the list
            new_item["objs"] = []
            for robj in refined_objs:
                if robj not in new_item["objs"]:
                    new_item["objs"].append(robj)

            new_caps.append(new_item)

        return new_caps

    def compute_chair(self, cap_file, sample_num, gpt_process=False, org_dir=None):

        '''
        Given ground truth objects and generated captions, determine which sentences have hallucinated words.
        '''

        self._load_generated_captions_into_evaluator(cap_file, sample_num, org_dir=org_dir)

        imid_to_objects = self.imid_to_objects
        caps = self.caps

        if gpt_process:
            caps, all_used_tokens, all_success_tokens = self.gpt_caption_processor()
            caps = self.get_pred_objs_match(caps)
        else:
            all_used_tokens = {}
            all_success_tokens = {}

        num_caps = 0.
        num_coco_caps = 0.
        num_hallucinated_caps = 0.
        hallucinated_word_count = 0.
        coco_word_count = 0.
        gt_word_count = 0.
        coco_obj_cls_count = 0.

        output = {'sentences': []}
        total_cap_word_num = 0
        for i, cap_eval in enumerate(caps):

            cap = cap_eval['caption']
            total_cap_word_num += len(cap.strip().split(" "))
            imid = cap_eval['image_id']

            #get all words in the caption, as well as corresponding node word
            if gpt_process:
                ext_objs = cap_eval["objs"]
                words, node_words, idxs = self.caption_objects_to_coco_objects(ext_objs)
                raw_words = ext_objs
            else:
                words, node_words, idxs, raw_words = self.caption_to_words(cap)

            gt_objects = imid_to_objects[imid]
            gt_word_count += len(gt_objects)
            cap_dict = {'image_id': cap_eval['image_id'],
                        'caption': cap, # org cap
                        'mscoco_hallucinated_words': [],
                        'mscoco_gt_words': list(gt_objects), # gt coco objs
                        'mscoco_generated_words': list(node_words), # gen mapped coco objs
                        'hallucination_idxs': [],
                        'words': raw_words # gpt process -> map double words -> postagging results, or original text words lemmas
                        }

            cap_dict['metrics'] = {'CHAIRs': 0,
                                   'CHAIRi': 0}

            #count hallucinated words, if [word, coco_obj_cls] is unique, count as one prediction
            coco_word_count += len(node_words)
            caption_coco_obj_cls = []

            hallucinated = False
            for word, node_word, idx in zip(words, node_words, idxs):
                if node_word[-1] not in gt_objects:
                    hallucinated_word_count += 1
                    cap_dict['mscoco_hallucinated_words'].append((word, node_word))
                    cap_dict['hallucination_idxs'].append(idx)
                    hallucinated = True
                else:
                    caption_coco_obj_cls.append(node_word[-1])

            caption_coco_obj_cls = set(caption_coco_obj_cls)
            # print(caption_coco_obj_cls)
            coco_obj_cls_count += len(caption_coco_obj_cls)

            #count hallucinated caps
            num_caps += 1
            if hallucinated:
               num_hallucinated_caps += 1

            cap_dict['metrics']['CHAIRs'] = int(hallucinated)
            cap_dict['metrics']['CHAIRi'] = 0.
            if len(words) > 0:
                num_coco_caps += 1
                cap_dict['metrics']['CHAIRi'] = len(cap_dict['mscoco_hallucinated_words'])/float(len(words))

            output['sentences'].append(cap_dict)

        chair_s = (num_hallucinated_caps/num_caps)
        chair_s_refine = (num_hallucinated_caps/num_coco_caps)
        chair_i = (hallucinated_word_count/coco_word_count)
        avg_word_len = float(total_cap_word_num)/num_caps
        obj_rec = coco_obj_cls_count/gt_word_count

        output['overall_metrics'] = {'CHAIRs': chair_s,
                                     'CHAIRs_refine': chair_s_refine,
                                     'CHAIRi': chair_i,
                                     'obj_rec': obj_rec,
                                     'sentence_num': num_caps,
                                     'coco_sentence_num': num_coco_caps,
                                     'coco_word_count': coco_obj_cls_count, # predict coco object classes
                                     'gt_word_count': gt_word_count, # ground truth coco object classes
                                     'avg_word_len': avg_word_len,
                                     'all_gpt_used_tokens': all_used_tokens,
                                     'all_gpt_success_tokens': all_success_tokens,
                                     'correct_rate': 1 - chair_s_refine,
                                     'object_correct_rate': 1 - chair_i
                                     }

        return output

def read_jsonl(jsonl_file):
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f1:
        for item in jsonlines.Reader(f1):
            data.append(item)
    return data

def load_generated_captions(cap_file, org_dir=None):
    if cap_file.endswith(".json"):
        #Read in captions
        caps = json.load(open(cap_file))
        try:
            metrics = caps['overall']
            caps = caps['imgToEval'].values()
            imids = set([cap['image_id'] for cap in caps])
        except:
            raise Exception("Expect caption file to consist of a dictionary with sentences correspdonding to the key 'imgToEval'")
    elif cap_file.endswith(".jsonl"):

        caps = read_jsonl(cap_file)

        if "image_id" not in caps[0].keys():
            try:
                assert org_dir != None and org_dir.strip() != ""
            except:
                raise Exception("Expect origin test input file directory for .jsonl cap file")
            cap_name = cap_file.split("/")[-1]
            org_name = cap_name.split("__")[0].replace("_answer", ".jsonl")

            if org_dir.endswith(".jsonl"):
                org_data_path = org_dir
            else:
                org_data_path = os.path.join(org_dir, org_name)
            org_data = read_jsonl(org_data_path)

        metrics = {}
        new_caps = []
        imids = []
        for i in range(len(caps)):
            if "image_id" not in caps[i].keys():
                imgid = int(org_data[i]["image_id"])
            else:
                imgid = int(caps[i]["image_id"])

            imids.append(imgid)

            if "prompt" in caps[i].keys():
                question = caps[i]["prompt"]
            elif "question" in caps[i].keys():
                question = caps[i]["question"]
            else:
                raise Exception("Expect 'question' or 'prompt' in generated file")

            if "text" in caps[i].keys():
                answer = caps[i]["text"].replace("Assistant:", "").strip()
            elif "answer" in caps[i].keys():
                answer = caps[i]["answer"].replace("Assistant:", "").strip()
            else:
                raise Exception("Expect 'answer' or 'text' in generated file")
            new_item = {"image_id": imgid, "question": question, "caption": answer}
            new_caps.append(new_item)
        caps = new_caps
        imids = set(imids)

    elif "." not in cap_file:
        caps = json.load(open(cap_file))
        try:
            assert 'raw_question' in caps[0].keys()
        except:
            raise Exception("Expect origin test input file directory for .jsonl cap file")
        imids = set([int(cap['question_id'].replace('.jpg')) for cap in caps])
        metrics = {}
        new_caps = []
        for item in caps:
            new_item = {
                "image_id": int(item["question_id"].replace('.jpg')),
                "question": item["raw_question"],
                "caption": item["answer"].replace("Assistant:", "").strip()
            }
            new_caps.append(new_item)
        caps = new_caps

    if "coh" in cap_file:
        caps = preprocess_coh_results(caps)

    return caps, imids, metrics

def save_hallucinated_words(cap_file, cap_dict, save_dir, sample_num):
    tag = cap_file.split('/')[-1].replace(".jsonl", "")
    with open(os.path.join(save_dir, f'hall_{tag}_{sample_num}.json'), 'w') as f:
        json.dump(cap_dict, f, indent=4)

def print_metrics(hallucination_cap_dict, quiet=False):
    sentence_metrics = hallucination_cap_dict['overall_metrics']
    metric_string = "%0.001f\t%0.001f\t%0.001f\t%d\t%d\t%0.01f" %(
                                        sentence_metrics['CHAIRs']*100,
                                        sentence_metrics['CHAIRs_refine']*100,
                                        sentence_metrics['CHAIRi']*100,
                                        sentence_metrics['sentence_num'],
                                        sentence_metrics['coco_sentence_num'],
                                        sentence_metrics['avg_word_len'])

    if not quiet:
        print("CHAIRs\tCHAIRsr\tCHAIRi\tsent_num\tcoco_num\tavg_len")
        print(metric_string)

    else:
        return metric_string

if __name__ == '__main__':
    time_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap_file", type=str, default='')
    parser.add_argument("--cap_folder", type=str, default='')
    parser.add_argument("--org_folder", type=str, default='')
    parser.add_argument("--cap_type", type=str, default='')
    parser.add_argument("--coco_path", type=str, default='./coco2014/annotations')
    parser.add_argument("--sample_num", type=int, default=-1)
    parser.add_argument("--use_gpt", action='store_true')
    parser.add_argument("--openai_key", type=str, default='')
    args = parser.parse_args()

    print("use gpt:", args.use_gpt)
    if args.cap_folder != '':
        patterns = ['*', '*/*', '*/*/*', '*/*/*/*']
        f_list = sum([list(glob.glob(args.cap_folder + p)) for p in patterns], [])
        cap_file_list_path = [x for x in f_list if x.endswith('.jsonl') and args.cap_type in x]
        random.shuffle(cap_file_list_path)
        args.cap_file = cap_file_list_path[0]
    else:
        cap_file_list_path = [args.cap_file]

    print("=======load prediction=======")
    print("load imgids file:", args.cap_file)
    _, imids, _ = load_generated_captions(args.cap_file, org_dir=args.org_folder)
    assert len(imids) == 300

    print("=======init evaluator=======")
    evaluator = CHAIR(imids, args.coco_path, args.openai_key)
    evaluator.get_annotations()

    print("========compute=========")
    for path in cap_file_list_path:
        print(path)
        tag = path.split('/')[-1].replace(".jsonl", "")

        save_dir = pathlib.Path(path).absolute().parent
        target_save_path_new = save_dir / f'hall_{tag}_{args.sample_num}.json'
        if target_save_path_new.exists():
            print("\teval file already exists!")
            continue
        else:
            print(f'Cannot find {target_save_path_new}')

        # if len(list(open(path))) != 300:
        #     continue

        temp_caps, temp_imids, _ = load_generated_captions(path, org_dir=args.org_folder)

        print("***do process***", flush=True)
        cap_dict = evaluator.compute_chair(path, args.sample_num, gpt_process=args.use_gpt, org_dir=args.org_folder)
        print(f'Done computing')

        save_hallucinated_words(path, cap_dict, save_dir, sample_num=args.sample_num)
        print(f'Done Saving')

        print_metrics(cap_dict)
        time_end = time.time()

        print("eval time:", time_end - time_start)