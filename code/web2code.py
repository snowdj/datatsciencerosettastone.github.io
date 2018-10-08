# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 18:30:11 2018

@author: oldyu
"""

from bs4 import BeautifulSoup
import requests

# when there are some . in the text line, better not to use \ end the line
# I had this problem - I eventually worked out that the reason was that I'd included \ characters in the string. If you have any of these, "escape" them with \\ and it should work fine.



pathtofile = r'G:\project\project1\sas\datatsciencerosettastone.github.io\code\\'


def get_code(language_type, pathtofile = r'G:\project\project1\sas\datatsciencerosettastone.github.io\code\\'):
    name = 'datasciencerosettastone'
    #language_type = 'sas'
    web_link = 'http://www.datasciencerosettastone.com/'
    page_link = web_link + language_type + '.html'
    # fetch the content from url
    page_response = requests.get(page_link, timeout=5)
    # parse html
    page_content = BeautifulSoup(page_response.content, "html.parser")
    
    class_name = language_type
    
    if language_type == 'R':
        class_name = 'sourceCode r'
    elif language_type == 'matlab':
        class_name = 'codeinput';
    # extract all html elements where price is stored
    code = page_content.find_all(class_=class_name)
    
    # you can also access the main_price class by specifying the tag of the class
    # sas_code1 = page_content.find_all('div', attrs={'class':'sas'})
    # pathtofile = 'G:\project\project1\sas\datatsciencerosettastone.github.io\code\'
    
    if language_type == 'sas':
        file_extension = 'sas'
    elif language_type == 'python':
        file_extension = 'py'
    elif language_type == 'R':
        file_extension = 'r'
    elif language_type == 'matlab':
        file_extension = 'm'
        
            
        
    filename = pathtofile + name + '.' + file_extension
    
    
    with open(filename, 'w') as myfile:  
        for i in code:
            myfile.write(i.text +'\n\n\n\n')
    


get_code(language_type = 'sas', pathtofile )

get_code(language_type = 'python', pathtofile )
    

get_code(language_type = 'R', pathtofile)
    

get_code('matlab', pathtofile)
    


page_link ='http://www.datasciencerosettastone.com/sas.html'
# fetch the content from url
page_response = requests.get(page_link, timeout=5)
# parse html
page_content = BeautifulSoup(page_response.content, "html.parser")

# extract all html elements where price is stored
sas_code = page_content.find_all(class_='sas')

# you can also access the main_price class by specifying the tag of the class
sas_code1 = page_content.find_all('div', attrs={'class':'sas'})


filename = 'G:\project\project1\sas\sas-prog-for-r-users\code\datasciencerosettastone.sas'


with open(filename, 'w') as myfile:  
    for i in sas_code:
        myfile.write(i.text +'\n\n\n\n')



