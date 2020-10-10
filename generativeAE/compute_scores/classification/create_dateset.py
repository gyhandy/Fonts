import pygame
import random
# from noises import NoiseUtils
import os
from tqdm import tqdm
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Attributes' values

# Background color
# Back_colors = {'red': (220, 20, 60), 'orange': (255, 165, 0), 'Yellow': (255, 255, 0), 'green': (0, 128, 0),
#           'cyan': (0, 255, 255),
#           'blue': (0, 0, 255), 'purple': (128, 0, 128), 'pink': (255, 192, 203), 'chocolate': (210, 105, 30),
#           'silver': (192, 192, 192)}
Colors = {'red': (220, 20, 60), 'orange': (255,165,0), 'Yellow': (255,255,0), 'green': (0,128,0), 'cyan' : (0,255,255),
         'blue': (0,0,255), 'purple': (128,0,128), 'pink': (255,192,203), 'chocolate': (210,105,30), 'silver': (192,192,192)}
# Sizes
Sizes = {'small': 80, 'medium': 100, 'large': 120}#456
# Fonts styles
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

# Words lower case and upper case
Letters = list(range(65, 91)) + list(range(97, 123))
Words = []
for i in Letters:
    Words.append(chr(i))

# location
# Locations = [(0, 0),(0, 1),(0, 2),
#              (1, 0),(1, 1),(1, 2),
#              (2, 0),(2, 1),(2, 2)]#only y axis
# Locations = {'up':(1, 0),'mid':(1, 1),'down':(1, 2)}
Locations = {'mid':(1, 1)}
# imgsize
img_size = 128
# save location
# font_dir = './'
font_dir = '/lab/tmpig23b/u/yao-data/dataset/classifiction_backcolor_center'
if not os.path.exists(font_dir):
    os.makedirs(font_dir)

pygame.init()
screen = pygame.display.set_mode((img_size, img_size))
for word in tqdm(Words): # 1st round for words
    for size in Sizes.keys():  # 2nd round for size
        for back_color in Colors.keys(): # 4th round for back_color
            for font_color in Colors.keys():  # 3rd round for font_color
                for font in All_fonts:  # 5th round for fonts
                    if font in useless_fonts:
                        continue
                    if font_color == back_color:
                        break
                    if random.random()<=0.2:
                        for location in Locations.keys():
                            try:
                                screen.fill(Colors[back_color])  # background color
                                selected_letter = word
                                selected_font = pygame.font.SysFont(font, Sizes[size], bold=True) # size and bold or not
                                font_size = selected_font.size(selected_letter);
                                rtext = selected_font.render(selected_letter, True, Colors[font_color], Colors[back_color])
                                drawX = img_size / 2 - (font_size[0] / 2.0) + (Locations[location][0]-1) * 0.8 * (img_size / 2 - (font_size[0] / 2.0))
                                drawY = img_size / 2 - (font_size[1] / 2.0) + (Locations[location][1]-1) * 0.8 * (img_size / 2 - (font_size[1] / 2.0))
                                screen.blit(rtext, (drawX, drawY))

                                img_name = selected_letter + '_' + size + '_' + font_color + '_' + back_color + '_' \
                                           + font +".png"
                                img_path = os.path.join(font_dir, 'test', back_color)
                                if not os.path.exists(img_path):
                                    os.makedirs(img_path)
                                pygame.image.save(screen, os.path.join(img_path, img_name))

                            except:
                                print(word, size, font_color, back_color, font)
                    else:
                        for location in Locations.keys():
                            try:
                                screen.fill(Colors[back_color])  # background color
                                selected_letter = word
                                selected_font = pygame.font.SysFont(font, Sizes[size],
                                                                    bold=True)  # size and bold or not
                                font_size = selected_font.size(selected_letter);
                                rtext = selected_font.render(selected_letter, True, Colors[font_color],
                                                             Colors[back_color])
                                drawX = img_size / 2 - (font_size[0] / 2.0) + (Locations[location][0] - 1) * 0.8 * (
                                            img_size / 2 - (font_size[0] / 2.0))
                                drawY = img_size / 2 - (font_size[1] / 2.0) + (Locations[location][1] - 1) * 0.8 * (
                                            img_size / 2 - (font_size[1] / 2.0))
                                screen.blit(rtext, (drawX, drawY))

                                img_name = selected_letter + '_' + size + '_' + font_color + '_' + back_color + '_' \
                                           + font + ".png"
                                img_path = os.path.join(font_dir, 'train', back_color)
                                if not os.path.exists(img_path):
                                    os.makedirs(img_path)
                                pygame.image.save(screen, os.path.join(img_path, img_name))

                            except:
                                print(word, size, font_color, back_color, font)
print('finished')