import itertools
import os
import re
import string
import sys
import json
import time
import pickle
import copy
import numpy as np
import pandas as pd
from collections import Counter
import concurrent.futures as futures
import tarfile

import sng_parser
import torch
import tqdm
import cv2
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import pycocotools.mask as mask_util
from pycocotools.coco import COCO

class BaseProcessor:
	def __init__(self):
		with open("reem/commom/tag_parser/ram_tag_list.txt", "r") as fr:
			tag_list = fr.readlines()
		self.tag_list = [tag.strip() for tag in tag_list]

	def visualize_ann(self, ann):
		return

	def process_wrapper(self, args):
		anns, idx, total_processes, task = args
		if task == "graph":
			task_func = self.parse_graph_single_process
		elif task == "tag":
			task_func = self.parse_tag_single_process
		else:
			raise ValueError(f"Task {task} not exists.")
		parsed_anns = []
		for ann in tqdm.tqdm(anns, desc=f"Process {idx + 1}/{total_processes}"):
			parsed_anns.append(task_func(ann))
		return parsed_anns

	def parse_tag_single_process(self, ann, key="dense"):
		graph = ann["extra_info"]["graph"]

		subject_tags = []
		object_tags = []
		entities = graph["entities"]
		relations = graph["relations"]
		objects = []
		for relation in relations:
			objects.append(relation["object"])
			object_tags.append(relation["relation"])
		objects = set(objects)
		subjects = set(list(range(len(entities)))) - objects

		for id in subjects:
			entity = entities[id]
			subject_tags.append(entity["head"])
			lemma_span = entity["lemma_span"]
			tokens = lemma_span.split(" ")
			for l in range(len(tokens)):
				tag = " ".join(tokens[-l-1:])
				subject_tags.append(tag)
			for modifier in entity["modifiers"]:
				if modifier["dep"] == "det":
					continue
				subject_tags.append(modifier["span"])

		for id in objects:
			entity = entities[id]
			object_tags.append(entity["head"])
			lemma_span = entity["lemma_span"]
			tokens = lemma_span.split(" ")
			for l in range(len(tokens)):
				tag = " ".join(tokens[-l - 1:])
				object_tags.append(tag)
			for modifier in entity["modifiers"]:
				if modifier["dep"] == "det":
					continue
				object_tags.append(modifier["span"])

		new_subject_tags = []
		for tag in set(subject_tags):
			if tag in self.tag_list:
				new_subject_tags.append(tag)

		new_object_tags = []
		for tag in set(object_tags):
			if tag in self.tag_list:
				new_object_tags.append(tag)

		subject_tag_idxs = [self.tag_list.index(el) for el in new_subject_tags]
		object_tag_idxs = [self.tag_list.index(el) for el in new_object_tags]

		ann["extra_info"]["subj_tags"] = subject_tag_idxs
		ann["extra_info"]["obj_tags"] = object_tag_idxs

		return ann

	def parse_graph_single_process(self, ann, key="dense"):
		caption = ann["caption"][key][0]
		graph = sng_parser.parse(caption)
		ann["extra_info"]["graph"] = graph
		return ann

	def parse_graph(self, file, max_workers=32):
		print(f"[INFO]:\tparse graph for ({file})")
		json_file = json.load(open(file, "r"))
		anns = json_file["annotations"]

		process_args = [(anns[i::max_workers], i, max_workers, "graph") for i in range(max_workers)]

		with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
			results = list(executor.map(self.process_wrapper, process_args))

		parsed_anns = []
		for result in results:
			parsed_anns.extend(result)

		json_file["annotations"] = parsed_anns
		print(f"[INFO]:\tdump annotation to ({file})")
		with open(file, "w") as fw:
			json.dump(json_file, fw)
		return file

	def parse_tags(self, file, max_workers=32):
		print(f"[INFO]:\tparse tags for ({file})")
		json_file = json.load(open(file, "r"))
		anns = json_file["annotations"]

		process_args = [(anns[i::max_workers], i, max_workers, "tag") for i in range(max_workers)]

		with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
			results = list(executor.map(self.process_wrapper, process_args))

		parsed_anns = []
		for result in results:
			parsed_anns.extend(result)

		json_file["annotations"] = parsed_anns
		print(f"[INFO]:\tdump annotation to ({file})")
		with open(file, "w") as fw:
			json.dump(json_file, fw)
		return file

class VGProcessor(BaseProcessor):
	def __init__(self, root="/home/ZhaoYuzhong/Dataset/lavis/vg/", version="vg1.2", save_name=None):
		self.VG_PATH = root
		self.UNK_IDENTIFIER = '<unk>'
		self.MAX_WORDS = 10
		self.version = version
		self.save_name = save_name
		format = {"info":{"description": None, "dataset_root": None, "extra_info": dict()},
				  "images":[{"id": None,
							 "file_name": None,
							 "height": None,
							 "width": None,
							 "extra_info": dict()
							 }],
				  "annotations":[{"id": None,
								  "image_id": None,
								  "segmentation": None,
								  "caption": None,
								  "extra_info": dict()}]}

		with open("rrecog/commom/tag_parser/ram_tag_list.txt", "r") as fr:
			tag_list = fr.readlines()
		self.tag_list = [tag.strip() for tag in tag_list]

		if version == "vg1.0":
			self.VG_IMAGE_ROOT = os.path.join("images")
			self.VG_ANN_ROOT = os.path.join(f"annotations/rrecog/vg1.0")
			self.VG_REGION_PATH = os.path.join(self.VG_PATH, f"annotations/vg1.0", "region_descriptions.json")
			self.VG_METADATA_PATH = os.path.join(self.VG_PATH, f"annotations/vg1.0", "image_data.json")
			self.VG_SPLIT_PATH = os.path.join(self.VG_PATH, f"annotations/vg1.0", "densecap_splits.json")
			self.vocabulary_size = 10000  # 10497#from dense caption paper
			self.HAS_VOCAB = True
		elif version == "vg1.2":
			self.VG_IMAGE_ROOT = os.path.join("images")
			self.VG_ANN_ROOT = os.path.join(f"annotations/rrecog/vg1.2")
			self.VG_REGION_PATH = os.path.join(self.VG_PATH, f"annotations/vg1.2", "region_descriptions.json")
			self.VG_METADATA_PATH = os.path.join(self.VG_PATH, f"annotations/vg1.2", "image_data.json")
			self.VG_SPLIT_PATH = os.path.join(self.VG_PATH, f"annotations/vg1.2", "densecap_splits.json")
			self.vocabulary_size = 10000  # 10497#from dense caption paper
			self.HAS_VOCAB = True
		else:
			self.VG_IMAGE_ROOT = os.path.join("images")
			self.VG_ANN_ROOT = os.path.join(f"annotations/rrecog/vgcoco")
			self.VG_REGION_PATH = os.path.join(self.VG_PATH, f"annotations/vg1.2", "region_descriptions.json")
			self.VG_METADATA_PATH = os.path.join(self.VG_PATH, f"annotations/vg1.2", "image_data.json")
			self.VG_SPLIT_PATH = os.path.join(self.VG_PATH, f"annotations/vg1.2", "densecap_splits.json")
			self.vocabulary_size = 10000  # 10497#from dense caption paper
			self.HAS_VOCAB = True

		if not os.path.exists(os.path.join(self.VG_PATH, self.VG_ANN_ROOT)):
			os.mkdir(os.path.join(self.VG_PATH, self.VG_ANN_ROOT))

		self.images_num = 0
		self.captions_num = 0
		self.skip_captions_num = 0
		self.num_invalid_bbox = 0
		self.num_empty_phrase = 0

	def init_vocabulary(self, phrases_all):
		words_to_count = {}
		word_freq = Counter(itertools.chain(*phrases_all))
		print(f"Found {len(word_freq.items())} unique word tokens.")
		vocab_freq = word_freq.most_common(self.vocabulary_size-1)
		self.vocabulary_inverted = [x[0] for x in vocab_freq]
		self.vocabulary_inverted.insert(0,self.UNK_IDENTIFIER)
		print(f"Using vocabulary size {self.vocabulary_size}.")
		print(f"The least frequent word in our vocabulary is '{vocab_freq[-1][0]}' and appeared {vocab_freq[-1][1]} times.")

	def dump_vocabulary(self, vocab_filename):
		print(f'Dumping vocabulary to file: {vocab_filename}')
		with open(vocab_filename, 'wb') as vocab_file:
			for word in self.vocabulary_inverted:
				vocab_file.write(f'{word}\n')
		print('Done.')

	def valid_image(self, image_info, regions, fix=True):
		image_info_h = image_info["height"]
		image_info_w = image_info["width"]
		image_id = image_info["id"]
		image_path = os.path.join(self.VG_IMAGE_ROOT, f"{image_id}.jpg")
		image_h, image_w, _ = cv2.imread(image_path).shape
		try:
			assert (image_h == image_info_h)
			assert (image_w == image_info_w)
			return True
		except:
			if fix:
				scale_x = image_w/image_info_w
				scale_y = image_h/image_info_h
				for region in regions:
					region["x"] = region["x"] * scale_x
					region["y"] = region["y"] * scale_y
					region["width"] = region["width"] * scale_x
					region["height"] = region["height"] * scale_y
			return False

	def vis_bbox(self, image_info, regions):
		image_id = image_info["id"]
		image_path = os.path.join(self.VG_IMAGE_ROOT, f"{image_id}.jpg")
		image = cv2.imread(image_path)
		image = torch.from_numpy(image)

		bboxes = []
		for region in regions:
			x1 = region["x"]
			y1 = region["y"]
			x2 = x1 + region["width"]
			y2 = y1 + region["height"]
			bbox = [x1, y1, x2, y2]
			bboxes.append(bbox)
		bboxes = torch.LongTensor(bboxes)
		image = draw_bounding_boxes(image.permute(2, 0, 1), bboxes, colors="yellow", width=3)

		fig, ax = plt.subplots()
		ax.imshow(image.permute(1, 2, 0).numpy())
		ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
		fig.tight_layout()

		fig.show()
		return

	def word_preprocess(self, phrase):
		""" preprocess a sentence: lowercase, clean up weird chars, remove punctuation """
		replacements = {
			u'½': u'half',
			u'—': u'-',
			u'™': u'',
			u'¢': u'cent',
			u'ç': u'c',
			u'û': u'u',
			u'é': u'e',
			u'°': u' degree',
			u'è': u'e',
			u'…': u'',
		}
		for k, v in replacements.items():
			phrase = phrase.replace(k, v)
		translation_table = str.maketrans("", "", string.punctuation)
		tokens = str(phrase).lower().translate(translation_table).split()
		return tokens

	def box_preprocess(self, obj, image_height, image_width):
		x1, y1, x2, y2 = obj["x"], obj["y"], obj["x"]+obj["width"], obj["y"]+obj["height"]
		# clamp to image
		valid = True
		if x1 < 1:
			x1 = 1
		if y1 < 1:
			y1 = 1
		if x1 > image_width - 1:
			x1 = image_width - 1
		if y1 > image_height - 1:
			y1 = image_height - 1
		if x2 > image_width:
			x2 = image_width
		if y2 > image_height:
			y2 = image_height
		# if x1 >= x2 and image_width - x1 > 0:
		# 	x2 = x1 + 1
		# 	valid = False
		# elif x1 >= x2 and x2 - 1 > 0:
		# 	x1 = x2 - 1
		# 	valid = False
		# if y1 >= y2 and image_height - y1 > 0:
		# 	y2 = y1 + 1
		# 	valid = False
		# elif y1 >= y2 and y2 - 1 > 0:
		# 	y1 = y2 - 1
		# 	valid = False

		try:
			assert x2 - x1 > 0
			assert y2 - y1 > 0
		except:
			valid = False

		segmentation = [[x1, y1, x1, y2, x2, y2, x2, y1]]
		return valid, segmentation

	def filter_images(self, regions_all, image_data, split_image_ids):
		filter_regions_all = []
		filter_image_data = []
		for img, idata in zip(regions_all, image_data):
			keep = img["id"] in split_image_ids and len(img['regions']) > 0
			if self.version == "vgcoco" and idata["coco_id"] is None:
				keep = False
			if keep:
				filter_regions_all.append(img)
				filter_image_data.append(idata)
		return filter_regions_all, filter_image_data

	def info_report(self):
		# Check the processing code, keep consistent with the offical code of DenseCap
		print(f"After filtering for splits there are {self.images_num} images")
		print(f"Keeping {self.captions_num} captions")
		print(f"Skipped {self.skip_captions_num} captions for being too long")
		print(f"Contain {self.num_empty_phrase} empty phrase")
		print(f"Contain {self.num_invalid_bbox} invalid bbox")
		return

	def tag_parser(self, phrase):
		subject_tags = []
		object_tags = []

		graph = sng_parser.parse(phrase)
		entities = graph["entities"]
		relations = graph["relations"]

		subjects = []
		objects = []
		for relation in relations:
			objects.append(relation["object"])
			object_tags.append(relation["relation"])
		objects = set(objects)
		subjects = set(list(range(len(entities)))) - objects

		for id in subjects:
			entity = entities[id]
			subject_tags.append(entity["head"])
			lemma_span = entity["lemma_span"]
			tokens = lemma_span.split(" ")
			for l in range(len(tokens)):
				tag = " ".join(tokens[-l-1:])
				subject_tags.append(tag)
			for modifier in entity["modifiers"]:
				if modifier["dep"] == "det":
					continue
				subject_tags.append(modifier["span"])

		for id in objects:
			entity = entities[id]
			object_tags.append(entity["head"])
			lemma_span = entity["lemma_span"]
			tokens = lemma_span.split(" ")
			for l in range(len(tokens)):
				tag = " ".join(tokens[-l - 1:])
				object_tags.append(tag)
			for modifier in entity["modifiers"]:
				if modifier["dep"] == "det":
					continue
				object_tags.append(modifier["span"])

		new_subject_tags = []
		for tag in set(subject_tags):
			if tag in self.tag_list:
				new_subject_tags.append(tag)

		new_object_tags = []
		for tag in set(object_tags):
			if tag in self.tag_list:
				new_object_tags.append(tag)

		subject_tag_idxs = [self.tag_list.index(el) for el in new_subject_tags]
		object_tag_idxs = [self.tag_list.index(el) for el in new_object_tags]

		return subject_tag_idxs, object_tag_idxs

	def process_dataset_offical(self, split_name, vocab=None):
		save_name = self.save_name if self.save_name is not None else split_name
		ann_root = os.path.join(self.VG_ANN_ROOT, f"{save_name}.json")
		save_path = os.path.join(self.VG_PATH, ann_root)

		dataset_sample = {"info":{"description": self.version,
								  "dataset_root": self.VG_PATH,
								  "image_root": self.VG_IMAGE_ROOT,
								  "annotation_root": ann_root,
								  "extra_info": dict()},
						  "images":[],
				   		  "annotations":[]}

		print("=" * 50)
		split = json.load(open(self.VG_SPLIT_PATH, "r"))
		split_image_ids = split[split_name]
		print(f'split image number: {len(split_image_ids)}')
		output_dataset_name = split_name
		print('start loading json files...')
		t1 = time.time()
		regions_all = json.load(open(self.VG_REGION_PATH))
		image_data = json.load(open(self.VG_METADATA_PATH))
		t2 = time.time()
		print(f'{(t2 - t1)} seconds for loading')

		regions_all, image_data = self.filter_images(regions_all, image_data, split_image_ids)

		num_bbox = 0
		num_empty_phrase = 0
		num_invalid_bbox = 0
		ann_id = 1
		for item, image_info in zip(tqdm.tqdm(regions_all), image_data):
			im_id = item['id']
			if self.version == "vg1.0":
				image_id = image_info["id"]
			else:
				image_id = image_info["image_id"]
			if im_id != image_id:
				print('region and image metadata inconsistent')
				exit()
			# tokenize phrase
			num_bbox += len(item['regions'])
			regions_filt = []

			image = f"{im_id}.jpg"
			abs_image_path = os.path.join(self.VG_PATH, self.VG_IMAGE_ROOT, image)
			assert os.path.exists(abs_image_path)
			image_h, image_w, _ = cv2.imread(abs_image_path).shape
			image_sample = {"id": im_id,
							"file_name": image,
							"height": image_h,
							"width": image_w,
					  		"extra_info": dict()}
			dataset_sample["images"].append(image_sample)

			for obj in item['regions']:
				phrase = obj['phrase']
				tokens = self.word_preprocess(phrase)
				if not len(tokens) <= 15:
					self.skip_captions_num += 1
					continue
				phrase = " ".join(tokens)

				if len(phrase) == 0:
					if split_name == "train":
						continue
					num_empty_phrase += 1

				regions_filt.append(obj)

				valid, segmentation = self.box_preprocess(obj, image_h, image_w)

				if not valid:
					if split_name == "train":
						continue
					num_invalid_bbox += 1

				subject_tags, object_tags = self.tag_parser(phrase)

				ann_sample = {"id": ann_id,
							  "image_id": im_id,
							  "caption": {"dense": [phrase]},
					   		  "segmentation": segmentation,
							  "extra_info": {"subj_tags": subject_tags,
											 "obj_tags": object_tags}}
				dataset_sample["annotations"].append(ann_sample)
				ann_id = ann_id + 1
		non_empty_image_ids = set([ann["image_id"] for ann in dataset_sample["annotations"]])
		images = dataset_sample["images"]
		filter_images = [image for image in images if image["id"] in non_empty_image_ids]
		dataset_sample["images"] = filter_images

		self.images_num = self.images_num + len(dataset_sample["images"])
		self.captions_num = self.captions_num + len(dataset_sample["annotations"])
		self.num_empty_phrase = self.num_empty_phrase + num_empty_phrase
		self.num_invalid_bbox = self.num_invalid_bbox + num_invalid_bbox

		print(f"[INFO]:\t{len(dataset_sample['annotations'])} annotations")
		print(f"[INFO]:\t{len(dataset_sample['images'])} images")
		print(f"[INFO]:\t{num_empty_phrase} empty phrase")
		print(f"[INFO]:\t{num_invalid_bbox} invalid bbox")
		print(f"[INFO]:\t({self.version}) dump annotation file to ({save_path})")
		with open(save_path, "w") as fw:
			json.dump(dataset_sample, fw)

	def process_dataset_coco(self, split_name, vocab=None):
		save_name = self.save_name if self.save_name is not None else split_name
		ann_root = os.path.join(self.VG_ANN_ROOT, f"{save_name}.json")
		save_path = os.path.join(self.VG_PATH, ann_root)

		dataset_sample = {"info":{"description": self.version,
								  "dataset_root": self.VG_PATH,
								  "image_root": self.VG_IMAGE_ROOT,
								  "annotation_root": ann_root,
								  "extra_info": dict()},
						  "images":[],
				   		  "annotations":[]}

		print("=" * 50)
		split = json.load(open(self.VG_SPLIT_PATH, "r"))
		split_image_ids = split[split_name]
		print(f'split image number: {len(split_image_ids)}')
		output_dataset_name = split_name
		print('start loading json files...')
		t1 = time.time()
		regions_all = json.load(open(self.VG_REGION_PATH))
		image_data = json.load(open(self.VG_METADATA_PATH))
		# regions_all = regions_all[:200]
		# image_data = image_data[:200]
		t2 = time.time()
		print(f'{(t2 - t1)} seconds for loading')

		phrases_all = []
		num_invalid_bbox = 0
		num_bbox = 0
		num_empty_phrase = 0
		ann_id = 1
		for item, image_info in zip(tqdm.tqdm(regions_all), image_data):
			if self.version == "vgcoco" and image_info["coco_id"] is None:
				continue
			im_id = item['id']
			if self.version == "vg1.0":
				image_id = image_info["id"]
			else:
				image_id = image_info["image_id"]
			if im_id != image_id:
				print('region and image metadata inconsistent')
				exit()
			if not im_id in split_image_ids:
				continue
			# tokenize phrase
			num_bbox += len(item['regions'])
			regions_filt = []

			# if not self.valid_image(image_info, item["regions"], fix=False):
			# 	continue
			# 	# self.vis_bbox(image_info, item["regions"])
			# 	num_invalid_image += 1

			image = f"{im_id}.jpg"
			abs_image_path = os.path.join(self.VG_PATH, self.VG_IMAGE_ROOT, image)
			assert os.path.exists(abs_image_path)
			image_h, image_w, _ = cv2.imread(abs_image_path).shape
			image_sample = {"id": im_id,
							"file_name": image,
							"height": image_h,
							"width": image_w,
					  		"extra_info": dict()}
			dataset_sample["images"].append(image_sample)

			for obj in item['regions']:

				# remove invalid regions
				if obj['x'] < 0 or obj['y'] < 0 or \
						obj['width'] <= 0 or obj['height'] <= 0 or \
						obj['x'] + obj['width'] >= image_info['width'] or \
						obj['y'] + obj['height'] >= image_info['height']:
					num_invalid_bbox += 1
					continue
				phrase = obj['phrase'].strip().encode('ascii', 'ignore').lower()

				# remove empty sentence
				if (len(phrase) == 0):
					num_empty_phrase += 1
					continue

				punctuation = string.punctuation.encode('ascii', 'ignore')

				obj['phrase_tokens'] = phrase.translate(None, punctuation).split()
				rrecog_phrase = phrase.translate(None, punctuation).decode("ascii")
				# remove regions with caption longer than max_words
				if len(obj['phrase_tokens']) > self.MAX_WORDS:
					continue
				regions_filt.append(obj)
				phrases_all.append(obj['phrase_tokens'])

				x1, y1, x2, y2 = obj["x"], obj["y"], obj["x"]+obj["width"], obj["y"]+obj["height"]
				segmentation = [[x1, y1, x1, y2, x2, y2, x2, y1]]

				ann_sample = {"id": ann_id,
							  "image_id": im_id,
							  "caption": {"dense": [rrecog_phrase]},
					   		  "segmentation": segmentation,
							  "extra_info": dict()}
				dataset_sample["annotations"].append(ann_sample)
				ann_id = ann_id + 1
		non_empty_image_ids = set([ann["image_id"] for ann in dataset_sample["annotations"]])
		images = dataset_sample["images"]
		filter_images = [image for image in images if image["id"] in non_empty_image_ids]
		dataset_sample["images"] = filter_images

		print(f"[INFO]:\t{len(dataset_sample['annotations'])} annotations")
		print(f"[INFO]:\t{len(dataset_sample['images'])} images")
		print(f"[INFO]:\t({self.version}) dump annotation file to ({save_path})")
		with open(save_path, "w") as fw:
			json.dump(dataset_sample, fw)

		if vocab is None:
			self.init_vocabulary(phrases_all)
		else:
			self.vocabulary_inverted = vocab
		self.vocabulary = {}
		for index, word in enumerate(self.vocabulary_inverted):
			self.vocabulary[word] = index

	def process(self):
		vocab = None
		# use existing vocabulary
		if self.HAS_VOCAB:
			vocab_path = os.path.join(self.VG_PATH, "annotations/vg1.0/vocabulary.txt")
			with open(vocab_path, 'r') as f:
				vocab = [line.strip() for line in f]

		datasets = ['train', 'val', 'test']
		for split_name in datasets:
			vocab = self.process_dataset_offical(split_name, vocab=vocab)

		self.info_report()

class VGREGProcessor(BaseProcessor):
	def __init__(self, root="/home/ZhaoYuzhong/Dataset/lavis/vg/", version="vg_reg", save_name=None):
		super().__init__()
		self.VG_PATH = root
		self.version = version
		self.save_name = save_name
		format = {"info":{"description": None, "dataset_root": None, "extra_info": dict()},
				  "images":[{"id": None,
							 "file_name": None,
							 "height": None,
							 "width": None,
							 "extra_info": dict()
							 }],
				  "annotations":[{"id": None,
								  "image_id": None,
								  "segmentation": None,
								  "caption": None,
								  "extra_info": dict()}]}

		self.VG_IMAGE_ROOT = os.path.join("images")
		self.VG_ANN_ROOT = os.path.join(f"annotations/reem/vg1.2")

		if not os.path.exists(os.path.join(self.VG_PATH, self.VG_ANN_ROOT)):
			os.mkdir(os.path.join(self.VG_PATH, self.VG_ANN_ROOT))

	def process(self):
		save_name = self.save_name if self.save_name is not None else "test_reg"
		ann_root = os.path.join(self.VG_ANN_ROOT, f"{save_name}.json")
		save_file = os.path.join(self.VG_PATH, ann_root)

		dataset_sample = {"info": {"description": self.version,
								   "dataset_root": self.VG_PATH,
								   "image_root": self.VG_IMAGE_ROOT,
								   "annotation_root": ann_root,
								   "extra_info": dict()},
						  "images": [],
						  "annotations": []}

		gt_path = os.path.join(self.VG_PATH, f"annotations/reem/vg1.2/glamm/test_caption.json")
		gt = COCO(gt_path)
		imgs = gt.imgs

		num_invalid_segmentation = 0
		for img_id, img in tqdm.tqdm(imgs.items()):
			image = img["file_name"]
			abs_image_path = os.path.join(self.VG_PATH, self.VG_IMAGE_ROOT, image)
			assert os.path.exists(abs_image_path)
			height, width = img["height"], img["width"]
			# image_h, image_w, _ = cv2.imread(abs_image_path).shape
			image_sample = {"id": img_id,
							"file_name": image,
							"height": height,
							"width": width,
							"extra_info": dict()}
			dataset_sample["images"].append(image_sample)

			anns = gt.imgToAnns[img_id]
			for ann in anns:
				bbox = ann["bbox"]
				x, y, w, h = bbox
				segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]

				ann_sample = {"id": ann["id"],
							  "image_id": img_id,
							  "caption": {"dense": [ann["caption"]]},
							  "segmentation": segmentation,
							  "extra_info": dict()}
				dataset_sample["annotations"].append(ann_sample)

		non_empty_image_ids = set([ann["image_id"] for ann in dataset_sample["annotations"]])
		images = dataset_sample["images"]
		filter_images = [image for image in images if image["id"] in non_empty_image_ids]
		dataset_sample["images"] = filter_images

		print(f"[INFO]:\tdataset ({self.version})")
		print(f"[INFO]:\t{len(dataset_sample['annotations'])} annotations")
		print(f"[INFO]:\t{len(dataset_sample['images'])} images")
		print(f"[INFO]:\tdump annotation to ({save_file})")
		with open(save_file, "w") as fw:
			json.dump(dataset_sample, fw)

		save_file = self.parse_graph(save_file)
		save_file = self.parse_tags(save_file)

class RefCOCOProcessor(BaseProcessor):
	def __init__(self, root="/home/ZhaoYuzhong/Dataset/lavis/refcoco/", version="refcoco"):
		super().__init__()

		self.root = root
		self.version = version
		self.ann_root = "annotations/reem/"

		if not os.path.exists(os.path.join(self.root, self.ann_root)):
			os.mkdir(os.path.join(self.root, self.ann_root))

		self.files = ["mdetr_annotations/finetune_refcocog_train.json",
					  "mdetr_annotations/finetune_refcocog_val_captions.json"]

		# self.files = ["mdetr_annotations/finetune_refcoco_train.json",
		# 			  "mdetr_annotations/finetune_refcoco+_train.json",
		# 			  "mdetr_annotations/finetune_refcocog_train.json",
		# 			  "mdetr_annotations/finetune_refcocog_val_captions.json"]

	def process_dataset_reem(self, file, save_file):
		if "refcoco" in self.version:
			self.image_root = "images/coco2017"
		else:
			self.image_root = "images/saiapr_tc-12/"

		dataset_sample = {"info": {"description": self.version,
								   "dataset_root": self.root,
								   "image_root": self.image_root,
								   "annotation_root": os.path.join(self.ann_root, os.path.basename(save_file)),
								   "extra_info": dict()},
						  "images": [],
						  "annotations": []}

		gt = COCO(file)
		imgs = gt.imgs

		for img_id, img in tqdm.tqdm(imgs.items()):
			image = img["file_name"].split("_")[-1]
			abs_image_path = os.path.join(self.root, self.image_root, image)
			assert os.path.exists(abs_image_path)
			height, width = img["height"], img["width"]
			image_sample = {"id": img_id,
							"file_name": image,
							"height": height,
							"width": width,
							"extra_info": dict()}
			dataset_sample["images"].append(image_sample)

			caption = img['caption']

			anns = gt.imgToAnns[img_id]
			for ann in anns:
				bbox = ann['bbox']
				x, y, w, h = bbox
				segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
				ann_sample = {"id": ann["id"],
							  "image_id": img_id,
							  "caption": {"dense": [caption]},
							  "segmentation": segmentation,
							  "extra_info": dict()}
				dataset_sample["annotations"].append(ann_sample)

		non_empty_image_ids = set([ann["image_id"] for ann in dataset_sample["annotations"]])
		images = dataset_sample["images"]
		filter_images = [image for image in images if image["id"] in non_empty_image_ids]
		dataset_sample["images"] = filter_images

		print(f"[INFO]:\t{len(dataset_sample['annotations'])} annotations")
		print(f"[INFO]:\t{len(dataset_sample['images'])} images")
		print(f"[INFO]:\t({self.version}) dump annotation file to ({save_file})")
		with open(save_file, "w") as fw:
			json.dump(dataset_sample, fw)

	def process(self):
		for file in self.files:
			save_file = os.path.basename(file).replace("finetune_", "").replace("_captions", "")
			file = os.path.join(self.root, self.ann_root, file)
			assert os.path.exists(file)
			save_file = os.path.join(self.root, self.ann_root, save_file)
			self.process_dataset_reem(file, save_file)
			self.parse_graph(save_file)
			self.parse_tags(save_file)

class Flickr30KProcessor(BaseProcessor):
	def __init__(self, root="/home/ZhaoYuzhong/Dataset/lavis/flickr30k/", version="flickr30k"):
		super().__init__()

		self.root = root
		self.version = version
		self.image_root = "images/train/"
		self.ann_root = os.path.join(f"annotations/reem/")

		if not os.path.exists(os.path.join(self.root, self.ann_root)):
			os.mkdir(os.path.join(self.root, self.ann_root))

		self.files = ["mdetr_annotations/final_flickr_mergedGT_train.json"]

	def process_dataset_reem(self, file, save_file):
		dataset_sample = {"info": {"description": self.version,
								   "dataset_root": self.root,
								   "image_root": self.image_root,
								   "annotation_root": os.path.join(self.ann_root, os.path.basename(save_file)),
								   "extra_info": dict()},
						  "images": [],
						  "annotations": []}

		gt = COCO(file)
		imgs = gt.imgs

		for img_id, img in tqdm.tqdm(imgs.items()):
			image = img["file_name"]
			abs_image_path = os.path.join(self.root, self.image_root, image)
			assert os.path.exists(abs_image_path)
			height, width = img["height"], img["width"]
			image_sample = {"id": img_id,
							"file_name": image,
							"height": int(height),
							"width": int(width),
							"extra_info": dict()}
			dataset_sample["images"].append(image_sample)

			caption = img['caption']
			# subject_tags, object_tags, graph = self.tag_parse(caption)

			anns = gt.imgToAnns[img_id]
			for ann in anns:
				bbox = ann['bbox']
				x, y, w, h = bbox
				segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
				ann_sample = {"id": ann["id"],
							  "image_id": img_id,
							  "caption": {"dense": [caption]},
							  "segmentation": segmentation,
							  "extra_info": dict()}
				dataset_sample["annotations"].append(ann_sample)

		non_empty_image_ids = set([ann["image_id"] for ann in dataset_sample["annotations"]])
		images = dataset_sample["images"]
		filter_images = [image for image in images if image["id"] in non_empty_image_ids]
		dataset_sample["images"] = filter_images

		print(f"[INFO]:\tdataset ({self.version})")
		print(f"[INFO]:\t{len(dataset_sample['annotations'])} annotations")
		print(f"[INFO]:\t{len(dataset_sample['images'])} images")
		print(f"[INFO]:\tdump annotation to ({save_file})")
		with open(save_file, "w") as fw:
			json.dump(dataset_sample, fw)

		return save_file

	def process(self):
		for file in self.files:
			save_file = "flickr30k_train.json"
			file = os.path.join(self.root, self.ann_root, file)
			assert os.path.exists(file)
			save_file = os.path.join(self.root, self.ann_root, save_file)
			save_file = self.process_dataset_reem(file, save_file)

			save_file = self.parse_graph(save_file)
			save_file = self.parse_tags(save_file)

# class GRIT20MProcessor(BaseProcessor):
# 	def __init__(self, root="/home/ZhaoYuzhong/Dataset/lavis/grit20m/", version="grit20m"):
# 		super().__init__()
#
# 		self.root = root
# 		self.version = version
# 		self.image_root = "images/"
# 		self.ann_root = os.path.join(f"annotations/reem/")
#
# 		if not os.path.exists(os.path.join(self.root, self.ann_root)):
# 			os.mkdir(os.path.join(self.root, self.ann_root))
#
# 		file_dir = "grit-20m"
# 		self.files = sorted([os.path.join(file_dir, file) for file in os.listdir(os.path.join(root, self.ann_root, file_dir))])
# 		self.save_files = [f"grit_train_{part}.json" for part in range(len(self.files))]
#
# 	def process_dataset_reem(self, file, save_file):
# 		dataset_sample = {"info": {"description": self.version,
# 								   "dataset_root": self.root,
# 								   "image_root": self.image_root,
# 								   "annotation_root": os.path.join(self.ann_root, os.path.basename(save_file)),
# 								   "extra_info": dict()},
# 						  "images": [],
# 						  "annotations": []}
#
# 		frames = pd.read_parquet(file)
#
# 		ann_id = 0
# 		for frame in tqdm.tqdm(frames.values):
# 			file_name = frame[0] + ".jpg"
# 			clip_score_vitb32 = frame[1]
# 			clip_score_vitl14 = frame[2]
# 			image_id = frame[3]
# 			url = frame[4]
# 			image_caption = frame[5]
# 			height = int(frame[6])
# 			width = int(frame[7])
# 			noun_chunks = frame[8]
# 			ref_exps = frame[9]
# 			assert len(noun_chunks) == len(ref_exps)
#
# 			image_sample = {"id": image_id,
# 							"file_name": file_name,
# 							"height": height,
# 							"width": width,
# 							"extra_info": {"clip_score_vitb32": clip_score_vitb32,
# 										   "clip_score_vitl14": clip_score_vitl14}}
# 			dataset_sample["images"].append(image_sample)
#
# 			for nc, re in zip(noun_chunks, ref_exps):
# 				noun = image_caption[int(nc[0]):int(nc[1])]
# 				noun_bbox = [nc[2] * width, nc[3] * height, nc[4] * width, nc[5] * height]
# 				noun_score = nc[6]
#
# 				caption = image_caption[int(re[0]):int(re[1])]
# 				bbox = [re[2] * width, re[3] * height, re[4] * width, re[5] * height]
# 				score = re[6]
#
# 				x1, y1, x2, y2 = bbox
# 				segmentation = [[x1, y1, x2, y1, x2, y2, x2, y1]]
# 				ann_sample = {"id": ann_id,
# 							  "image_id": image_id,
# 							  "caption": {"dense": {"caption": caption, "score": score}, "noun": {"caption": noun, "score": noun_score}},
# 							  "segmentation": segmentation,
# 							  "extra_info": dict()}
# 				dataset_sample["annotations"].append(ann_sample)
# 				ann_id = ann_id + 1
#
# 		non_empty_image_ids = set([ann["image_id"] for ann in dataset_sample["annotations"]])
# 		images = dataset_sample["images"]
# 		filter_images = [image for image in images if image["id"] in non_empty_image_ids]
# 		dataset_sample["images"] = filter_images
#
# 		print(f"[INFO]:\tdataset ({self.version})")
# 		print(f"[INFO]:\t{len(dataset_sample['annotations'])} annotations")
# 		print(f"[INFO]:\t{len(dataset_sample['images'])} images")
# 		print(f"[INFO]:\tdump annotation to ({save_file})")
# 		with open(save_file, "w") as fw:
# 			json.dump(dataset_sample, fw)
#
# 		return save_file
#
# 	def process(self):
# 		for file, save_file in zip(self.files, self.save_files):
# 			file = os.path.join(self.root, self.ann_root, file)
# 			save_file = os.path.join(self.root, self.ann_root, save_file)
# 			assert os.path.exists(file)
# 			save_file = self.process_dataset_reem(file, save_file)
#
# 			# save_file = self.parse_graph(save_file)
# 			# save_file = self.parse_tags(save_file)
#
# 	def extract_images(self):
# 		tar_dir = os.path.join(self.root, "GRIT")
# 		save_dir = os.path.join(self.root, self.image_root)
#
# 		files = [os.path.join(tar_dir, file) for file in os.listdir(tar_dir) if file.endswith('.tar')]
#
#
# 		for file in tqdm.tqdm(files):
# 			try:
# 				output_dir = os.path.join(save_dir, os.path.basename(os.path.splitext(file)[0]))
# 				if not os.path.exists(output_dir):
# 					os.mkdir(output_dir)
# 				with tarfile.open(file, 'r') as tar:
# 					tar.extractall(output_dir)
# 			except:
# 				print(f"Invalid tar file ({file})")
#
# 		extracted_images = [os.path.join(save_dir, file) for file in os.listdir(save_dir) if file.endswith("jpg")]
# 		extracted_jsons = [os.path.join(save_dir, file) for file in os.listdir(save_dir) if file.endswith("json")]
# 		extracted_txts = [os.path.join(save_dir, file) for file in os.listdir(save_dir) if file.endswith("txt")]
# 		# for file in tqdm.tqdm(extracted_jsons + extracted_txts):
# 		# 	os.remove(file)
#
# 		print(f"[INFO]:\textract {len(extracted_images)} images")
# 		return
#
# 	def gather_annotations(self):
# 		save_file = os.path.join(self.root, self.ann_root, "grit_train.json")
# 		image_dir = os.path.join(self.root, self.image_root)
# 		exist_images = set([image for image in os.listdir(image_dir) if image.endswith("jpg")])
# 		dataset_sample = {"info": {"description": self.version,
# 								   "dataset_root": self.root,
# 								   "image_root": self.image_root,
# 								   "annotation_root": os.path.join(self.ann_root, os.path.basename(save_file)),
# 								   "extra_info": dict()},
# 						  "images": [],
# 						  "annotations": []}
# 		ann_id = 0
# 		for file in tqdm.tqdm(self.save_files):
# 			file = os.path.join(self.root, self.ann_root, file)
# 			assert os.path.exists(file)
# 			json_file = COCO(file)
#
# 			valid_images = set([el["file_name"] for el in json_file.imgs.values()]).intersection(exist_images)
#
# 			for image_id, image_sample in tqdm.tqdm(json_file.imgs.items()):
# 				if image_sample["file_name"] == "000239986.jpg":
# 					dataset_sample["images"].append(image_sample)
# 					ann_samples = json_file.imgToAnns[image_id]
# 					for ann_sample in ann_samples:
# 						ann_sample["id"] = ann_id
# 						dataset_sample["annotations"].append(ann_sample)
# 						ann_id = ann_id + 1
#
# 		non_empty_image_ids = set([ann["image_id"] for ann in dataset_sample["annotations"]])
# 		images = dataset_sample["images"]
# 		filter_images = [image for image in images if image["id"] in non_empty_image_ids]
# 		dataset_sample["images"] = filter_images
#
# 		print(f"[INFO]:\tdataset ({self.version})")
# 		print(f"[INFO]:\t{len(dataset_sample['annotations'])} annotations")
# 		print(f"[INFO]:\t{len(dataset_sample['images'])} images")
# 		print(f"[INFO]:\tdump annotation to ({save_file})")
# 		with open(save_file, "w") as fw:
# 			json.dump(dataset_sample, fw)
#
# 		return

class GRIT20MProcessor(BaseProcessor):
	def __init__(self, root="/home/ZhaoYuzhong/Dataset/lavis/grit20m/", version="grit20m"):
		super().__init__()

		self.root = root
		self.version = version
		self.image_root = "images/"
		self.ann_root = os.path.join(f"annotations/reem/")

		if not os.path.exists(os.path.join(self.root, self.ann_root)):
			os.mkdir(os.path.join(self.root, self.ann_root))

		self.file = os.path.join(self.root, "GRIT")
		self.save_file = os.path.join(self.root, self.ann_root, "grit_train.json")

	def process_dataset_reem(self, file, save_file):
		dataset_sample = {"info": {"description": self.version,
								   "dataset_root": self.root,
								   "image_root": self.image_root,
								   "annotation_root": os.path.join(self.ann_root, os.path.basename(save_file)),
								   "extra_info": dict()},
						  "images": [],
						  "annotations": []}

		frames = pd.read_parquet(file)

		ann_id = 0
		for frame in tqdm.tqdm(frames.values):
			file_name = frame[0] + ".jpg"
			clip_score_vitb32 = frame[1]
			clip_score_vitl14 = frame[2]
			image_id = frame[3]
			url = frame[4]
			image_caption = frame[5]
			height = int(frame[6])
			width = int(frame[7])
			noun_chunks = frame[8]
			ref_exps = frame[9]
			assert len(noun_chunks) == len(ref_exps)

			image_sample = {"id": image_id,
							"file_name": file_name,
							"height": height,
							"width": width,
							"extra_info": {"clip_score_vitb32": clip_score_vitb32,
										   "clip_score_vitl14": clip_score_vitl14}}
			dataset_sample["images"].append(image_sample)

			for nc, re in zip(noun_chunks, ref_exps):
				noun = image_caption[int(nc[0]):int(nc[1])]
				noun_bbox = [nc[2] * width, nc[3] * height, nc[4] * width, nc[5] * height]
				noun_score = nc[6]

				caption = image_caption[int(re[0]):int(re[1])]
				bbox = [re[2] * width, re[3] * height, re[4] * width, re[5] * height]
				score = re[6]

				x1, y1, x2, y2 = bbox
				segmentation = [[x1, y1, x2, y1, x2, y2, x2, y1]]
				ann_sample = {"id": ann_id,
							  "image_id": image_id,
							  "caption": {"dense": {"caption": caption, "score": score}, "noun": {"caption": noun, "score": noun_score}},
							  "segmentation": segmentation,
							  "extra_info": dict()}
				dataset_sample["annotations"].append(ann_sample)
				ann_id = ann_id + 1

		non_empty_image_ids = set([ann["image_id"] for ann in dataset_sample["annotations"]])
		images = dataset_sample["images"]
		filter_images = [image for image in images if image["id"] in non_empty_image_ids]
		dataset_sample["images"] = filter_images

		print(f"[INFO]:\tdataset ({self.version})")
		print(f"[INFO]:\t{len(dataset_sample['annotations'])} annotations")
		print(f"[INFO]:\t{len(dataset_sample['images'])} images")
		print(f"[INFO]:\tdump annotation to ({save_file})")
		with open(save_file, "w") as fw:
			json.dump(dataset_sample, fw)

		return save_file

	def recursive_search(self, sdir):
		imgs = []
		anns = []
		for root, dirs, files in os.walk(sdir):
			# txts.extend(os.path.join(root, file) for file in files if file.endswith("txt"))
			anns.extend(os.path.join(root, file) for file in files if file.endswith("json"))
			imgs.extend(os.path.join(root, file) for file in files if file.endswith("jpg"))
		return imgs, anns

	def process(self):
		imgs, anns = self.recursive_search(os.path.join(self.root, self.image_root))
		return

	def extract_images(self):
		tar_dir = os.path.join(self.root, "GRIT")
		save_dir = os.path.join(self.root, self.image_root)

		files = [os.path.join(tar_dir, file) for file in os.listdir(tar_dir) if file.endswith('.tar')]


		for file in tqdm.tqdm(files):
			try:
				output_dir = os.path.join(save_dir, os.path.basename(os.path.splitext(file)[0]))
				if not os.path.exists(output_dir):
					os.mkdir(output_dir)
				with tarfile.open(file, 'r') as tar:
					tar.extractall(output_dir)
			except:
				print(f"Invalid tar file ({file})")

		extracted_images = [os.path.join(save_dir, file) for file in os.listdir(save_dir) if file.endswith("jpg")]
		extracted_jsons = [os.path.join(save_dir, file) for file in os.listdir(save_dir) if file.endswith("json")]
		extracted_txts = [os.path.join(save_dir, file) for file in os.listdir(save_dir) if file.endswith("txt")]
		# for file in tqdm.tqdm(extracted_jsons + extracted_txts):
		# 	os.remove(file)

		print(f"[INFO]:\textract {len(extracted_images)} images")
		return

	def gather_annotations(self):
		save_file = os.path.join(self.root, self.ann_root, "grit_train.json")
		image_dir = os.path.join(self.root, self.image_root)
		exist_images = set([image for image in os.listdir(image_dir) if image.endswith("jpg")])
		dataset_sample = {"info": {"description": self.version,
								   "dataset_root": self.root,
								   "image_root": self.image_root,
								   "annotation_root": os.path.join(self.ann_root, os.path.basename(save_file)),
								   "extra_info": dict()},
						  "images": [],
						  "annotations": []}
		ann_id = 0
		for file in tqdm.tqdm(self.save_files):
			file = os.path.join(self.root, self.ann_root, file)
			assert os.path.exists(file)
			json_file = COCO(file)

			valid_images = set([el["file_name"] for el in json_file.imgs.values()]).intersection(exist_images)

			for image_id, image_sample in tqdm.tqdm(json_file.imgs.items()):
				if image_sample["file_name"] == "000239986.jpg":
					dataset_sample["images"].append(image_sample)
					ann_samples = json_file.imgToAnns[image_id]
					for ann_sample in ann_samples:
						ann_sample["id"] = ann_id
						dataset_sample["annotations"].append(ann_sample)
						ann_id = ann_id + 1

		non_empty_image_ids = set([ann["image_id"] for ann in dataset_sample["annotations"]])
		images = dataset_sample["images"]
		filter_images = [image for image in images if image["id"] in non_empty_image_ids]
		dataset_sample["images"] = filter_images

		print(f"[INFO]:\tdataset ({self.version})")
		print(f"[INFO]:\t{len(dataset_sample['annotations'])} annotations")
		print(f"[INFO]:\t{len(dataset_sample['images'])} images")
		print(f"[INFO]:\tdump annotation to ({save_file})")
		with open(save_file, "w") as fw:
			json.dump(dataset_sample, fw)

		return


if __name__ == "__main__":
	refcoco = RefCOCOProcessor()
	refcoco.process()
	# processor = GRIT20MProcessor()
	# processor.process()
	# processor.process()
	# processor.extract_images()
	# processor.gather_annotations()
	# processor = Flickr30KProcessor()
	# processor.process()

