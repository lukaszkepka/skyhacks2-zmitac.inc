import os
import re
from random import random, randint
from textgenrnn import textgenrnn
from glob import glob
import json

datasets_folder = 'D:\\Datasets\\skyhacks\\additional_task_data\\input'
models_folder = 'D:\\Programowanie\\zmitac.inc\\Models'


def load_model():
    # return textgenrnn()
    return textgenrnn(weights_path=os.path.join(models_folder, 'colaboratory_weights4.hdf5'),
                      vocab_path=os.path.join(models_folder, 'colaboratory_vocab4.json'),
                      config_path=os.path.join(models_folder, 'colaboratory_config4.json'))


def post_process_generated_text(text):
    return text.replace('\n', '')

def generate_pdf():
    import sys, string, os
    os.system("D:\\Programowanie\\zmitac.inc\\Brochure\\HtmlToPdf\\HtmlToPdf\\bin\\Debug\\HtmlToPdf.exe")

def generate_text_for_image(file_name, description, category, text_generator, n=5):
    text = ""
    j = 0

    for i in description:
        keyword = list(i.values())[0]
        score = list(i.values())[1]

        generated_text = text_generator.generate(n=5, prefix=keyword, temperature=[1.0, 0.5, 0.2, 0.2], return_as_list=True,
                                                 max_gen_length=100)
        generated_text = post_process_generated_text(generated_text[randint(0, 4)])
        generated_text = re.sub("\s\s+", " ", generated_text)
        text = text + generated_text + ". "
        j += 1

        if n == j:
            break

    print(text + '\n')
    return text


def run():
    pdf_generation_json = {}
    text_generator = load_model()
    for test_dir in glob(datasets_folder + '\\*'):
        pdf_generation_json[os.path.basename(test_dir)] = []
        test_dir_path = os.path.join(datasets_folder, test_dir)
        summary_file = os.path.join(test_dir_path, 'summary.txt')
        with open(summary_file, "r") as f:
            for line in f.readlines():
                a = line.split(',')
                file_name = a[0]
                json_file = a[0] + '.json'
                category = a[1].strip()

                with open(os.path.join(test_dir_path, json_file), 'r') as f:
                    json_desc = json.load(f)

                text = generate_text_for_image(file_name, json_desc, category, text_generator)
                pdf_generation_json[os.path.basename(test_dir)].append({'text': text, 'image': file_name})

    with open('D:\\Programowanie\\zmitac.inc\\Brochure\\HtmlToPdf\\HtmlToPdf\\bin\\Debug\\data.json', 'w') as outfile:
        json.dump(pdf_generation_json, outfile)

    generate_pdf()


run()
