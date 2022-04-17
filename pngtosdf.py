import requests
import os

'''
This script transfer molecule image to mol format (saved as sdf files).
You should change the image folder (specified in line 11) ane the name of result file (specified in line 26).
'''

def get_sdf(name, img_folder):

    imgs76 = os.listdir(img_folder)

    url = 'https://molvec.ncats.io/molvec'
    headers = {'Content-Type' : 'image/jpg'}

    for imgs in imgs76:
        with open('{}/{}'.format(img_folder, img), 'rb') as fp:
            r = requests.post(url, data=fp, headers=headers)

        with open('{}.sdf'.format(patent), 'a') as fp:
            fp.write(r.json()['molvec']['molfile'])
            fp.write('\n$$$$\n')

def main():
    name = 'abc'
    img_folder = "***"
    get_sdf(name, img_folder)

if __name__ == '__main__':
    main()
