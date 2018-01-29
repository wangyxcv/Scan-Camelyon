import matplotlib.pyplot as plt
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log', nargs='?', type=str, help="log file")
args = parser.parse_args()


loss = []
x = []
y = []
with open(args.log,"r") as f:
    lines = f.readlines()
	#print type(lines)
    pattern = "Train net output #0: loss = \d+\.\d+"
    #pattern_1 = "Iteration \d+"
    txt = "".join(lines).replace("\n","")
	#print len(txt),type(txt)
    loss = re.findall(pattern, txt)
    #iter = re.findall(pattern_1, txt)
    for l in loss:
		#y.append(re.findall("\d+", l)[0])
        x.append(float(re.findall("\d+\.\d+", l)[0]))
    y = [i*20 for i in range(len(x))]
    plt.plot(y, x, 'g--')
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(args.log.split("/")[-1])
	#plt.axis([0,100000,0,3])
    plt.show()
	#print x
	#print y
