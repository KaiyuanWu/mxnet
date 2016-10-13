#for no additional label
#example: python test_verify_data_iter.py generate_img_list /tmp/mnist/train/ /tmp/mnist/train_list.txt 1
#         
def generate_img_list(data_dir, listfilename, shuffle = True):
    persons = os.listdir(data_dir)
    id_map = dict(enumerate(persons))
    listfile = open(listfilename,'w')
    for id in id_map.keys():
	person = id_map[id]
	img_dir = data_dir + "/"  + person
	imgs = os.listdir(img_dir)
	count = 0
	for img in imgs:
	    listfile.write("%010d %s\n"%(id, img_dir+"/"+img))
	    if count > 100:
		break
	    count += 1
    listfile.close()
    if shuffle:
	cmd = "cat "+ listfilename+" | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > "+listfilename+".r"
	os.system(cmd)
	cmd = "mv "+ listfilename+".r "+listfilename
	os.system(cmd)
