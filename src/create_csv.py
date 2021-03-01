import sys 
import os.path
if __name__ == "__main__":
if len(sys.argv) != 2:
    print "usage: create_csv <base_path>" 
    sys.exit(1)

BASE_PATH=sys.argv[1] 
SEPARATOR=";"
label = 0 
for dirname, dirnames, filenames in os.walk(BASE_PATH): 
    for subdirname in dirnames: 
        subject_path = os.path.join(dirname, subdirname) 
        for filename in os.listdir(subject_path): 
            if "female" in subject_path: 
                    label = 0 
            else: 
                label = 1 
            abs_path = "%s/%s" % (subject_path, filename) 
            print "%s%s%d" % (abs_path, SEPARATOR, label)