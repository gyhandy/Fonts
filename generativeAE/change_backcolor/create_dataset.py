#encoding: utf-8
'''
Create a font dataset for trainding
Content / size / color(Font) / color(background) / style
E.g. A / 64/ red / blue / arial
potential : position (x, y) bold, rotation

'''
import os
import pygame
from pygame.locals import *
# wd2 = 'ad ah ai am an as at ba be by do ed ee em en er ex fe fu go ha he hi id if in is it ko la li ma me mm mu my na no of oh ok on oo op or os pa pi qi re so ta to uh um un up ur us vu we wo xi xu ye yo zo'
# wd3 = 'the, and, for, are, but, not, you, all, any, can, had, her, was, one, our, out, day, get, has, him, his, how, man, new, now, old, see, two, way, who, boy, did, its, let, put, say, she, too, use'

'''reference'''
# color 10 (back ground and font)
Colors = {'red': (220, 20, 60), 'orange': (255,165,0), 'Yellow': (255,255,0), 'green': (0,128,0), 'cyan' : (0,255,255),
         'blue': (0,0,255), 'purple': (128,0,128), 'pink': (255,192,203), 'chocolate': (210,105,30), 'silver': (192,192,192)}
# size 3
Sizes = {'small': 80, 'medium' : 100, 'large': 120}
# Sizes = {'small': 20, 'medium': 40, 'large': 60}
# style nearly over 100
All_fonts = pygame.font.get_fonts()
useless_fonts = ['notocoloremoji', 'droidsansfallback', 'gubbi', 'kalapi', 'lklug',  'mrykacstqurn', 'ori1uni','pothana2000','vemana2000',
                'navilu', 'opensymbol', 'padmmaa', 'raghumalayalam', 'saab', 'samyakdevanagari']
useless_fontsets = ['kacst', 'lohit', 'sam']
# throw away the useless
for useless_font in useless_fonts:
    try:
        All_fonts.remove(useless_font)
    except:
        pass

temp = All_fonts.copy()
for useless_font in temp: # check every one
    for set in useless_fontsets:
        if set in useless_font:
            try:
                All_fonts.remove(useless_font)
            except:
                print(useless_font)


All_fonts = All_fonts[:10]

# letter 52
Letters = list(range(65, 68)) + list(range(97, 100))
img_size = 128


font_dir = '/lab/tmpig23b/u/yao-data/dataset/change_backcolor'
if not os.path.exists(font_dir):
    os.makedirs(font_dir)

try:
    os.environ["DISPLAY"]
except:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
screen = pygame.display.set_mode((img_size, img_size)) # image size Fix(128 * 128)


for letter in Letters: # 1st round for letters
    for size in Sizes.keys():  # 2nd round for size
        for font_color in Colors.keys():  # 3rd round for font_color
            for back_color in Colors.keys():  # 4th round for back_color
                # if not back_color == font_color:''' should not be same '''
                for font in All_fonts:  # 5th round for fonts
                    if not font_color == back_color:
                        try:
                            # 1 set back_color
                            screen.fill(Colors[back_color]) # background color
                            # 2 set letter
                            selected_letter = chr(letter)
                            # 3,4 set font and size
                            selected_font = pygame.font.SysFont(font, Sizes[size]) # size and bold or not
                            # 5 set font_color
                            rtext = selected_font.render(selected_letter, True, Colors[font_color], Colors[back_color])
                            font_size = selected_font.size(selected_letter);
                            drawX = img_size / 2 - (font_size[0] / 2.0)
                            drawY = img_size / 2 - (font_size[1] / 2.0)
                            # screen.blit(rtext, (img_size/2, img_size/2))
                            # screen.blit(rtext, (img_size / 4, 0))
                            # screen.blit(rtext, (10, 0)) # because
                            screen.blit(rtext, (drawX, drawY))
                            # E.g. A / 64/ red / blue / arial
                            img_name = selected_letter + '_' + size + '_' + font_color + '_' + back_color + '_' + font + ".png"
                            img_path = os.path.join(font_dir, back_color)
                            if not os.path.exists(img_path):
                                os.makedirs(img_path)
                            pygame.image.save(screen, os.path.join(img_path, img_name))
                        except:
                            print(letter, size, font_color, back_color, font)
                    else:
                        break