import json
with open("data//slink//ang.asc") as f:
    ang = f.read()
with open("data//slink//bbox.txt","r")as f:
    bbox = json.loads(f.read())
ang = ang.splitlines()
print(ang)
cols = int(ang[0].split("\t")[1])
rows = int(ang[1].split("\t")[1])
xllcorner = float(ang[2].split("\t")[1])
yllcorner = float(ang[3].split("\t")[1])
cellsize = float(ang[4].split("\t")[1])
no_data = float(ang[5].split("\t")[1])


ang = ang[6:]
arr = [item.split("\t") for item in ang]
for i in range(len(arr)):
    arr[i] = [float(item) for item in arr[i][:-1]]




dct = {}
for y in range(rows):
    for x in range(cols):
        dct[str(x)+"_"+str(y)] = float(arr[y][x])#\*88

xscaling = {}
x_scale = abs(bbox["east"] - bbox["west"])/cols
x_start = bbox["west"]
for x in range(cols):
    low = x_start+x*x_scale
    high = x_start+(x+1)*x_scale
    xscaling[str(x)] = [low,high]


yscaling = {}
y_scale = abs(bbox["south"] - bbox["north"])/rows
y_start = bbox["north"]
for y in range(rows):
    low = y_start-y*y_scale
    high = y_start-(y+1)*y_scale
    yscaling[str(y)] = [low,high]

scaling = {}
scaling["x"] = xscaling
scaling["y"] = yscaling

with open("output//ang.txt","w") as f:
    json.dump(dct,f)
with open("output//scaling.txt","w") as f:
    json.dump(scaling,f)