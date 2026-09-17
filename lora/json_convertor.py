import json
import os




def json_to_text(file, save_dir=''):
    # file = 

    with open(file, 'r') as f:
        data = [json.loads(line) for line in f]
        # print(data)
    file_base_name = os.path.dirname(file)
    for item in data:
        # filename = d0['file_name']
        # text = d0['text']
        
        img_filename = item['file_name']
        text = item['text']
        
        base_name = os.path.splitext(img_filename)[0]
        txt_filename = f"{base_name}.txt"
        
        if save_dir:
            output_path = os.path.join(save_dir, txt_filename)
        else:
            output_path = os.path.join(file_base_name, txt_filename)
            
            # save_dir이 비어있으면 jsonl 파일이 있는 디렉토리에 저장
            # output_path = os.path.join(os.path.dirname(file_path), txt_filename)

        # 텍스트 파일 저장
        with open(output_path, 'w', encoding='utf-8') as out_f:
            out_f.write(text.strip())
    print(f"Converted {len(data)} items from {file} to text files.")

json_to_text(
    '/workspace/dataset/myimg/img/metadata.jsonl'
)