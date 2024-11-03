import json
import argparse
import pandas as pd
from collections import defaultdict

def jsonl_excel(args, text_prompts):

    category_score_win ={
        'Coarse Perception': 0,
        'Fine-grained perception': 0,
        'Relation reasoning': 0,
        'Attribute reasoning': 0,
        'Time series inference': 0,
        'Mechanical logical reasoning': 0,
        'Creative generation': 0,
        'OCR': 0
    }

    category_score_loss ={
        'Coarse Perception': 0,
        'Fine-grained perception': 0,
        'Relation reasoning': 0,
        'Attribute reasoning': 0,
        'Time series inference': 0,
        'Mechanical logical reasoning': 0,
        'Creative generation': 0,
        'OCR': 0
    }

    category_score_tie ={
        'Coarse Perception': 0,
        'Fine-grained perception': 0,
        'Relation reasoning': 0,
        'Attribute reasoning': 0,
        'Time series inference': 0,
        'Mechanical logical reasoning': 0,
        'Creative generation': 0,
        'OCR': 0
    }

    category_model_loss = defaultdict(int)
    category_model_win = defaultdict(int)
    category_model_tie = defaultdict(int)


    count_win = 0
    count_loss = 0
    count_tie  = 0

    for i in range(len(text_prompts)):
        category_name = text_prompts[i]['type_name']

        modelA = json.dumps(text_prompts[i]['modelA'])

        if text_prompts[i]['score'] == 1:
            category_score_win[category_name] += 1
            category_model_win[modelA] +=1
            count_win += 1
        elif text_prompts[i]['score']  + 1 == 0:
            category_score_loss[category_name] += 1
            category_model_loss[modelA] +=1
            count_loss += 1
        else:
            category_score_tie[category_name] +=1
            category_model_tie[modelA] +=1
            count_tie  += 1

    count_excel = {}
    count_excel['model A'] = text_prompts[0]['modelA']
    count_excel['model B'] = text_prompts[0]['modelB']
    count_excel['win'] = count_win
    count_excel['loss'] = count_loss
    count_excel['tie'] = count_tie
    count_excel['score'] = (count_win + (count_tie)/2) / (count_win + count_loss + count_tie)

    model_key = list(category_model_win.keys())[0] if len(category_model_win) > 0 else list(category_model_tie.keys())[0]
    count_excel['model B win to model A'] = category_model_win[model_key]
    count_excel['model B loss to model A'] = category_model_loss[model_key]

    count_excel['Coarse Perception'] = str(category_score_win['Coarse Perception'])  +'/' + str(category_score_loss['Coarse Perception']) +'/' + str(category_score_tie['Coarse Perception'])
    count_excel['Fine-grained perception'] = str(category_score_win['Fine-grained perception']) +'/' +  str(category_score_loss['Fine-grained perception'])  +'/' + str(category_score_tie['Fine-grained perception'])
    count_excel['Relation reasoning']= str(category_score_win['Relation reasoning'])  +'/' + str(category_score_loss['Relation reasoning']) +'/' + str(category_score_tie['Relation reasoning'])
    count_excel['Attribute reasoning']= str(category_score_win['Attribute reasoning']) +'/' +  str(category_score_loss['Attribute reasoning'])  +'/' +  str(category_score_tie['Attribute reasoning'])
    count_excel['Time series inference']= str(category_score_win['Time series inference']) +'/' +  str(category_score_loss['Time series inference'])  +'/' + str(category_score_tie['Time series inference'])
    count_excel['Mechanical logical reasoning']= str(category_score_win['Mechanical logical reasoning'])  +'/' +  str(category_score_loss['Mechanical logical reasoning'])+'/' +  str(category_score_tie['Mechanical logical reasoning'])
    count_excel['Creative generation']= str(category_score_win['Creative generation'])  +'/' +    str(category_score_loss['Creative generation'])    +'/' + str(category_score_tie['Creative generation'])
    count_excel['OCR']= str(category_score_win['OCR'])  +'/' +    str(category_score_loss['OCR'] )+'/' + str(category_score_tie['OCR'])

    count_excel['WIN Check']= sum(category_score_win.values())
    count_excel['LOSS Check']= sum(category_score_loss.values())
    count_excel['TIE Check']= sum(category_score_tie.values())
    count_excel['ALL Check']= sum(category_score_win.values()) +  sum(category_score_loss.values())+ sum(category_score_tie.values())


    for key, value in count_excel.items():
        if not isinstance(value, list):
            count_excel[key] = [value]

    df1 = pd.DataFrame(count_excel)
    path1 = args.text_prompt
    with pd.ExcelWriter(path1+'.xlsx', engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='Sheet1', index=False)


def jsonl_excel_all_data(args, text_prompts):
    for i in range(len(text_prompts)):
        text_prompts[i]['image_path_list'] = text_prompts[i]['image_path_list']
        text_prompts[i]['type_name'] = text_prompts[i]['type_name']

        a= text_prompts[i]['prompt'].index("[Beginning of Model A's answer]") + len("[Beginning of Model A's answer]")
        b = text_prompts[i]['prompt'].index("[End of Model A\'s answer]")
        c= text_prompts[i]['prompt'].index("[Beginning of Model B's answer]") + len("[Beginning of Model B's answer]")
        d = text_prompts[i]['prompt'].index("[End of Model B\'s answer]")

        description_begin = text_prompts[i]['prompt'].index("[Beginning of the detailed description of the picture]") + len("[Beginning of the detailed description of the picture]")
        description_end = text_prompts[i]['prompt'].index("[End of the detailed description of the picture]")

        question_begin = text_prompts[i]['prompt'].index("[Beginning of the user's question]") + len("[Beginning of the user's question]")
        question_end = text_prompts[i]['prompt'].index("[End of the user's question]")

        text_prompts[i]['model A answer']= text_prompts[i]['prompt'][a : b]
        text_prompts[i]['model B answer']= text_prompts[i]['prompt'][c : d]

        text_prompts[i]['question']=text_prompts[i]['prompt'][question_begin : question_end]
        text_prompts[i]['description']=text_prompts[i]['prompt'][description_begin : description_end]
        text_prompts[i]['prompt'] = ' '

    df = pd.DataFrame(text_prompts)
    path = args.text_prompt
    df.to_excel(path +'_all_data'+'.xlsx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RefoMB evaluation')
    parser.add_argument('--text_prompt', type=str,
                            default='Omnilmm_answers_sampled_base.jsonl')
    parser.add_argument('--get_all_data', action='store_true')
    args = parser.parse_args()

    with open(args.text_prompt ,encoding='utf8') as f:
        text_prompts = json.load(f)

    jsonl_excel(args, text_prompts)
    if args.get_all_data:
        jsonl_excel_all_data(args, text_prompts)
