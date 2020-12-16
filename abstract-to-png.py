import sys
import gzip
import time
from wand.color import Color
from wand.image import Image
from wand.drawing import Drawing
from wand.compat import nested
from textwrap import wrap
import threading

def word_wrap(image, ctx, text, roi_width, roi_height):
    """Break long text to multiple lines, and reduce point size
    until all text fits within a bounding box."""
    mutable_message = text
    iteration_attempts = 100

    def eval_metrics(txt):
        """Quick helper function to calculate width/height of text."""
        metrics = ctx.get_font_metrics(image, txt, True)
        return (metrics.text_width, metrics.text_height)

    while ctx.font_size > 0 and iteration_attempts:
        iteration_attempts -= 1
        width, height = eval_metrics(mutable_message)
        if height > roi_height:
            ctx.font_size -= 0.75  # Reduce pointsize
            mutable_message = text  # Restore original text
        elif width > roi_width:
            columns = len(mutable_message)
            while columns > 0:
                columns -= 1
                mutable_message = '\n'.join(wrap(mutable_message, columns))
                wrapped_width, _ = eval_metrics(mutable_message)
                if wrapped_width <= roi_width:
                    break
            if columns < 1:
                ctx.font_size -= 0.75  # Reduce pointsize
                mutable_message = text  # Restore original text
        else:
            break
    if iteration_attempts < 1:
        raise RuntimeError("Unable to calculate word_wrap for " + text)
    return mutable_message

ROI_SIDE = 256
FONT_SIZE = 18

def render_text(fname, message, font):
    with Drawing() as draw:
        with Image(width=256, height=32, background=Color('white')) as img:
            draw.font_family = font
            draw.font_size = FONT_SIZE
            draw.fill_color = Color('black')
            # mutable_message = word_wrap(img, draw, message, ROI_SIDE, ROI_SIDE)
            draw.text(0, FONT_SIZE, message)
            draw(img)
            img.save(filename=fname)


N_IMAGES = 200
# FONTS = ["Arial", "Times New Roman", "Comic Sans", "Courier New", "Calibri", "Candara", "Consolas", "Georgia", "Corbel", "Arial Black"]

FONTS = [
      "Arial", 
      "Times New Roman", 
      "Courier New", 
      "Calibri", 
      "Candara", 
      "Georgia", 
      "Corbel",
      "Helvetica",
      "Comic Sans MS",
      "Garamond"]

total_count_lock = threading.Lock()
total_count = 0

def worker(lines, start, end):
    global total_count

    counter = start
    fname = "test-data-1/{number} {font}.png"

    for i in range(start, end):
        
        for font in FONTS:
            render_text(fname.format(number=counter, font=font), lines[i][:500], font)

        counter += 1

        total_count_lock.acquire()
        total_count += 1

        if total_count % 10 == 0:
            print("processed: {images} time: {time}".format(images=total_count, time=time.process_time()))
        total_count_lock.release()

lines = []

with gzip.open('abstract-per-line.txt.gz', 'rt', encoding="utf-8") as ingzip:
    lines = ingzip.readlines()

# threading.Thread(target=worker,args=(lines, 1000, 1999),).start()
# threading.Thread(target=worker,args=(lines, 2000, 2999),).start()
# threading.Thread(target=worker,args=(lines, 3000, 3999),).start()
# threading.Thread(target=worker,args=(lines, 4000, 4999),).start()

threading.Thread(target=worker,args=(lines, 0, 99),).start()
threading.Thread(target=worker,args=(lines, 100, 199),).start()
