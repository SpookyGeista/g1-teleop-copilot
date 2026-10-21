"""Generate synthetic browser-test pixels, not simulation evidence."""
from PIL import Image, ImageDraw
from pathlib import Path
import os
p=Path(os.environ.get('ASSIST_TEST_OUTPUT','/tmp/assist_frontend_test'));p.mkdir(exist_ok=True,parents=True)
im=Image.new('RGB',(640,360),(24,32,39));d=ImageDraw.Draw(im)
d.rounded_rectangle((145,42,465,318),radius=9,fill=(78,89,99),outline=(130,143,152),width=3)
for x in [160,448]:
    for y in [58,302]:d.ellipse((x-4,y-4,x+4,y+4),fill=(170,182,190))
for x in [210,335]:
    d.rectangle((x,145,x+70,215),fill=(28,34,38),outline=(170,180,185),width=3)
    d.rounded_rectangle((x+22,158,x+49,201),radius=4,fill=(179,191,197))
d.text((181,83),'FRONTEND TEST FIXTURE',fill=(200,213,220))
d.text((207,239),'INPUT A       INPUT B',fill=(205,218,224))
im.save(p/'fixture.jpg')
