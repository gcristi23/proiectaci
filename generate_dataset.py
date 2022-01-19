from PIL import Image
import os
import numpy as np

def get_number(name):
    return name.split('.')[0][-3:]

if __name__ == '__main__':
    window_size = 200
    N_crops = 10

    base_dir = './dataset/text'
    dataset = os.listdir(base_dir)

    outputs_base = './gt/text'
    outputs = {get_number(x): x for x in os.listdir(outputs_base)}

    save_path = "./data/{}/text/{}"

    for image_name in dataset:
        image_path = os.path.join(base_dir, image_name)
        image = Image.open(image_path)

        width,height = image.size   
        
        n = get_number(image_name)
        output_path = os.path.join(outputs_base, outputs[n])
        output = Image.open(output_path)
        N_crops = ((width * height)//(window_size**2))+2
        for i in range(N_crops):
            x = np.random.randint(0, width-window_size-1)
            y = np.random.randint(0, height-window_size-1)

            
            new_crop = output.crop((x,y,x+window_size,y+window_size))
            file_name = outputs[n].split('.')[0]+'.jpg'
            output_save_path = save_path.format("output",'a'+str(i)+file_name)
            new_crop.save(output_save_path, 'jpeg')

            new_crop = image.crop((x,y,x+window_size,y+window_size))
            file_name = image_name.split('.')[0]+'.jpg'
            new_crop.save(save_path.format("input",str(i)+file_name), 'jpeg')

