# NAI
# Generate key file looking at top level of UCF directory (e.g. UCF11_updated_mpg)
# This only needs to be done once as the key file should not change unless adding another class
# The output of this file is a txt based key file that contains the mappings of english word labels
#   to integer labels (i.e. 'shooting 0\ngolf 1\n...')
import os
import glob

# root of the UCF dataset
root_ucf_directory = os.path.join(os.path.expanduser('~'),"DukeML/datasets/UCF11/UCF11_updated_mpg")

# where to save key file
key_file = os.path.join(os.path.expanduser('~'),"DukeML/datasets/UCF11/KEYFILE_UCF11.txt")

# open the keyfile for writing
f = open(key_file, 'wb')

category_count = 0

# for each subdirectory in root dir [ex. UCF11_updated_mpg/basketball]
for dir1 in glob.glob(root_ucf_directory + '/*'):

	# for each subdirectory in new category level directory [ex. UCF11_updated_mpg/basketball/v_shooting_01]
	for dir2 in glob.glob(dir1 + "/*"):

		# ex. tmp = v_shooting_01
		tmp = os.path.split(dir2)[-1]

		# split tmp on "_" and extract the second item, which is the category
		# ex. category = shooting
		category = tmp.split("_")[1]

		output = category + " " + str(category_count)
		print output

		# write the output to the file
		f.write(output + "\n")

		# increment category count so each category has unique id
		category_count += 1

        # break, only needed to look at one
		break

f.close()
