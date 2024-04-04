import random
import shutil
import os

def sampling(source_url, destination_url, source_boundary) :
    sampling_tbl = [i for i in range(source_boundary)]
    for i in range(0, 500) :
        t = random.randint(i, source_boundary - 1)
        sampling_tbl[i], sampling_tbl[t] = sampling_tbl[t], sampling_tbl[i]
        
        file_name = os.path.join(source_url, f'{sampling_tbl[i]:06d}.jpg')
        dest_name = os.path.join(destination_url, f'{i:06d}.jpg')
        print(file_name, dest_name)
        try :
            shutil.move(file_name, dest_name)
        except Exception as e :
            print(e)

if __name__ == '__main__' :
    sampling('D:\\captured_pics', 'D:\\training_datas\\training_image_samples', 857)
