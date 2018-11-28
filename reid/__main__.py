# import segmentation.example_resnet_binary
from termcolor import colored
import sys
import os

sys.path.append('.')

from segmentation import network as segnet


'''
        1) Take 2 inputs (sys.argv): path to one image file (query) and one directory (storing gallery of images)
        2) Run the segmentation network on the entire gallery, store the results in a new directory
                note: before running the segmentation network on the gallery, check if the new directory already exists.  Make a naming convention
        3) Run the segmentation network on the query image.  Store it only in memory.

        To do steps 2) and 3), you would need a function that runs the segmentation network and takes the batch size as input.

        4) Now, run the autoencoder (look at testCAE.py) on the entire gallery and store the embeddings (obtained from model.embedding) for all images in memory (preferable) or on disk (if memory is not enough).
        5) Run the autoencoder (without decoder) on the query image and store the embedding in memory

                You can remove the decoder by doing this: model.decoder = nn.Sequential()

        6) Write a function that takes k as input and returns the k closest embeddings of images in
           the gallery to the embedding of the query image.  (can write an algorithm using max heaps
           and dictionaries to do that; even a partial sort would work; or there may be a built in 
           pytorch function, though this is unlikely)

        7) Find the corresponding images to the closest embeddings.  (don't make a dictionary whose 
           keys are embeddings please!, find a less memory intensive solution). You should have k 
           images after this

        8) Try to make a graphic like Figure 5 of this paper (https://arxiv.org/pdf/1711.08565.pdf).
           If you can't get the red/green dots, it is fine.  But hopefully, everything else should be
           easy to do.  Note that this graphic should be one image on its own.  Display this to the 
           user/save it to tensorboard


           This may take a good bit of time, if you don't organize the code well.  Before starting to
           code, you shoud try whiteboarding the functions/classes you intend on using.  Hopefully, 
           that'll make life much simpler.  The earlier we can finish this the better.

           Note that you'll have to also ask the user to get the weights file to both the segmenation
           and CAE network
'''

def main():
    pass

def display_usage():
    print(colored('Usage:\n', 'red', attrs=['bold', 'underline']) + colored('python3 reid /path/to/queryimage.jpg /directory/of/gallery/images /directory/to/output/to' , 'red'))


# args:
# query image filename, gallery images directory, output directory
if __name__ == '__main__':
    PROPER_NUM_ARGS = 4
    args = sys.argv
    if(len(args) != PROPER_NUM_ARGS):
        display_usage()
        sys.exit(0)
        
        

    query_path = sys.argv[1]
    gallery_image_dir = sys.argv[2]
    output_dir = sys.argv[3]
    output_dir_query = os.path.join(output_dir, 'segmented_query/')
    output_dir_gallery = os.path.join(output_dir, 'segmented_gallery/')

    multiple_queries = False
    
    if(os.path.isdir(query_path)):
        multiple_queries = True
    elif(os.path.isfile(query_path)):
        pass
    else:
        print(colored('Could not locate Query Image directory or file. Please see usage' , 'red'))
        display_usage()
        sys.exit(0)
    if(not os.path.isdir(gallery_image_dir)):
        print(colored('Could not locate gallery directory. Please see usage' , 'red'))
        display_usage()
        sys.exit(0)
    
    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir_query):
        os.makedirs(output_dir_query)
    if not os.path.exists(output_dir_gallery):
        os.makedirs(output_dir_gallery)
   
                
    # SEGMENTATION NETWORK
    
    segnet.process_images(query_path, output_dir_query,  batchsize=1)
    
    print('Segmented Query Image and outputted to: ' + str(os.path.join(output_dir, '/segmented_query/')))
    segnet.process_images(gallery_image_dir, output_dir_gallery , batchsize=1)


    # CAE
    


    print(gallery_image_dir)
    print()
    print(colored('-- ABR-Net --', 'blue', attrs=['bold']))
    main()

