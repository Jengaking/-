import random
import shutil
import os

def sampling(source_url, destination_url, start_number, last_frame_number) :
    width = last_frame_number - start_number + 1
    sample_size = 500
    sampling_tbl = [i for i in range(width)]
    for i in range(0, sample_size) :
        t = random.randint(i,  width - 1)
        sampling_tbl[i], sampling_tbl[t] = sampling_tbl[t], sampling_tbl[i]
        
        file_name = os.path.join(source_url, f'{sampling_tbl[i] + start_number:06d}.jpg')
        dest_name = os.path.join(destination_url, f'{i+500:06d}.jpg')
        print(file_name, dest_name)
        try :
            shutil.move(file_name, dest_name)
        except Exception as e :
            print(e)

if __name__ == '__main__' :
    sampling('D:\\captured_pics', 'D:\\training_datas\\training_image_samples',start_number = 630, last_frame_number= 2107)
