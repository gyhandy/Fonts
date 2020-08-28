#encoding: utf-8
'''
Create a font dataset for trainding
Content / size / color(Font) / color(background) / style
E.g. A / 64/ red / blue / arial
potential : position (x, y) bold, rotation

'''
import os
import pygame
import numpy as np

'''reference'''
# color 10 (back ground and font)
Colors = {'red': (220, 20, 60), 'Yellow': (255,255,0), 'green': (0,128,0), 'cyan' : (0,255,255),
         'blue': (0,0,255), 'chocolate': (210,105,30), 'pink': (255,192,203)}
# discarded colors
# 'purple': 'orange': (255,165,0), (128,0,128),   , 'silver': (192,192,192)
# size 3
# Sizes = {'small': 80, 'medium' : 100, 'large': 120}
# Sizes = {'small': 40, 'medium': 50, 'large': 60}
Sizes = {'small': 45, 'large': 60}
# style nearly over 100
All_fonts = pygame.font.get_fonts()
useless_fonts = ['notocoloremoji', 'droidsansfallback', 'gubbi', 'kalapi', 'lklug',  'mrykacstqurn', 'ori1uni','pothana2000','vemana2000',
                'navilu', 'opensymbol', 'padmmaa', 'raghumalayalam', 'saab', 'samyakdevanagari', 'arplukaihk', 'arplukaitw', 'arplukaitwmbe',
                 'arpluminghk', 'arplumingtw', 'arplumingtwmbe','chandas','dejavusansmono', 'dejavuserif', 'gargi', 'freesans',
                 'jamrul', 'khmerossystem', 'liberationsans', 'liberationsansnarrow', 'notosanscjkhkblack', 'notosanscjkhkdemilight',
                 'notosanscjkhklight', 'notosanscjkhkmedium', 'notosanscjkjpblack', 'notosanscjkjpdemilight', 'notosanscjkjpmedium',
                 'notosanscjkjplight', 'notosanscjkkrdemilight', 'notosanscjkkrblack','notosanscjkkrmedium', 'notosanscjkkrlight','notosanscjksc',
                 'notosanscjkscblack', 'notosanscjkscdemilight', 'notosanscjksclight', 'notosanscjkscmedium', 'notosanscjkscthin', 'notosanscjktc',
                 'notosanscjktcblack', 'notosanscjktcdemilight', 'notosanscjktclight', 'notosanscjktcmedium', 'notosanscjktcthin',
                 'notoserifcjksc', 'notoserifcjktc', 'padmaa', 'rekha', 'tlwgtypewriter', 'tlwgtypo', 'ubuntucondensed',
                 'ubuntumono','aakar','anjalioldlipi','chandas', 'chilanka', 'dyuthi','keraleeyam','khmeros','khmerossystem',
                 'kinnari','manjari','meera','nakula', 'rachana','suruma', 'tibetanmachineuni','uroob']
useless_fontsets = ['kacst', 'lohit', 'sam']
# throw away the useless
for useless_font in useless_fonts:
    # print(useless_font)
    try:
        All_fonts.remove(useless_font)
    except:
        print(useless_font)
temp = All_fonts.copy()
for useless_font in temp: # check every one
    for set in useless_fontsets:
        if set in useless_font:
            try:
                All_fonts.remove(useless_font)
            except:
                print(useless_font)
# letter 52
Letters = list(range(65, 91)) + list(range(97, 123))
Integers = list(range(48, 58))
# words = []
img_size = 128
# generate words

def sample_words():
    rs = []
    for i in range(26):
        # 1
        rs.append(chr(np.random.randint(65, 91)))
        # 2
        rs.append(chr(np.random.randint(65, 91)) + chr(np.random.randint(65, 91)))
        # 3
        rs.append(chr(np.random.randint(65, 91)) + chr(np.random.randint(65, 91)) + chr(np.random.randint(65, 91)))
    return rs

def dfs(w, L):
    if len(w)==3:
        return
    for l in L:
        words.append(w+chr(l))
        dfs(w+chr(l), L)


# dfs('', Letters + Integers)
words = sample_words()

font_dir = '/home2/fonts_dataset_version0upper'
if not os.path.exists(font_dir):
    os.makedirs(font_dir)

pygame.init()
screen = pygame.display.set_mode((img_size, img_size)) # image size Fix(128 * 128)
cnt = 0

# for letter in Letters: # 1st round for letters
for word in words: # 1st round for words
    print(word)
    for size in Sizes.keys():  # 2nd round for size
        print(size)
        for font_color in Colors.keys():  # 3rd round for font_color
            for back_color in Colors.keys():  # 4th round for back_color
                # if not back_color == font_color:''' should not be same '''
                for font in All_fonts:  # 5th round for fonts
                    if not font_color == back_color:
                        cnt +=1
                        print(cnt, '/288288')
                        try:
                            # 1 set back_color
                            screen.fill(Colors[back_color]) # background color
                            # 2 set letter/word
                            # selected_letter = chr(letter)
                            selected_letter = word
                            # 3,4 set font and size
                            selected_font = pygame.font.SysFont(font, Sizes[size]) # size and bold or not
                            font_size = selected_font.size(selected_letter);
                            # 5 set font_color

                            rtext = selected_font.render(selected_letter, True, Colors[font_color], Colors[back_color])
                            # 6 render
                            drawX = img_size / 2 - (font_size[0] / 2.0)
                            drawY = img_size / 2 - (font_size[1] / 2.0)
                            # screen.blit(rtext, (img_size/2, img_size/2))
                            # screen.blit(rtext, (img_size / 4, 0))
                            screen.blit(rtext, (drawX, drawY)) # because
                            # E.g. A / 64/ red / blue / arial
                            img_name = selected_letter + '_' + size + '_' + font_color + '_' + back_color + '_' + font + ".png"
                            img_path = os.path.join(font_dir, selected_letter, size, font_color, back_color, font)
                            if not os.path.exists(img_path):
                                os.makedirs(img_path)
                            pygame.image.save(screen, os.path.join(img_path, img_name))
                        except:
                            # print(letter, size, font_color, back_color, font)
                            print(word, size, font_color, back_color, font)
                    else:
                        break








# screen.fill((255,255,255)) # background color
# start, end = (97, 255) # 汉字编码范围
# for codepoint in range(int(start), int(end)):
#     word = chr(codepoint)
#     font = pygame.font.SysFont("arial", 64) # size and bold or not
#     # font = pygame.font.Font("msyh.ttc", 64)
#     rtext = font.render(word, True, (0, 0, 0), (255, 255, 255))
#     # pygame.image.save(rtext, os.path.join(chinese_dir, word + ".png"))
#     screen.blit(rtext, (300, 300))
#     pygame.image.save(screen, os.path.join(chinese_dir, word + ".png"))
