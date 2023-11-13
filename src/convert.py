import os
import re
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely._utils import camel_to_snake
from supervisely.io.fs import file_exists, get_file_name, get_file_name_with_ext
from tqdm import tqdm

import src.settings as s


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    dataset_path = "/home/grokhi/rawdata/cassed/CaSSed_Dataset_Final/real_world_data"
    batch_size = 50

    tag_names = ["browns_field", "main_trail", "powerline"]

    def get_ann_path(path):
        parent = os.path.dirname(path)
        grandparent = os.path.dirname(parent)
        if "raw_" in parent:
            if "Dataset1B - Powerline/" in path:
                return f"{grandparent}/annotations/{get_file_name_with_ext(path)}"
            elif "Main_Trail/" in path:
                ann_name = get_file_name_with_ext(path).split("screenshot_")[-1]
                return f"{grandparent}/annotations/{ann_name}"
            elif "Dataset1A-Brown_field/" in path:
                ann_name = f"rgb{get_file_name_with_ext(path).split('image')[-1]}"
                return f"{grandparent}/annotations/{ann_name}"
            elif "Dataset2_Fogdata_Segmentation/" in path:
                return f"{grandparent}/annotations/{get_file_name_with_ext(path)}"
            elif "Dataset3_NorthFarm_Segmentation/" in path:
                return f"{grandparent}/annotations/{get_file_name_with_ext(path)}"
            elif "Dataset4_NorthSlope_Segmentation/Dataset2" in path:
                ann_name = get_file_name(path)
                return f"{grandparent}/annotations/{ann_name}.png"
        elif "imgs" in parent:
            ann_name = f"anno_{get_file_name_with_ext(path).split('img_')[-1]}"
            return f"{grandparent}/annos/{ann_name}"

    def create_ann(image_path):
        labels, tag_sly = [], []
        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]
        ann_path = get_ann_path(image_path)

        for tag in tag_names:
            if tag in image_path.lower() and "/Test" in image_path:
                tag_sly = [sly.Tag(tag_meta) for tag_meta in tag_metas if tag_meta.name == tag]
        if file_exists(ann_path):
            mask_np = sly.imaging.image.read(ann_path)

            for rgb, cls in rgbs.items():
                r, g, b = rgb
                mask = (mask_np[:, :, 0] == r) & (mask_np[:, :, 1] == g) & (mask_np[:, :, 2] == b)
                mask_np[:, :, :3][mask] = [labels2cls[cls], labels2cls[cls], labels2cls[cls]]

            mask_np = mask_np[:, :, 0]

            if len(np.unique(mask_np)) != 1:
                uniq = [elem for elem in list(np.unique(mask_np)) if elem in classes.keys()]
                for label in uniq:
                    if label != 0:
                        obj_mask = mask_np == label
                        curr_bitmap = sly.Bitmap(obj_mask)
                        obj_class = meta.get_obj_class(classes[label])
                        curr_label = sly.Label(curr_bitmap, obj_class)
                        labels.append(curr_label)
        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tag_sly)

    obj_smooth_trail = sly.ObjClass("smooth trail", sly.Bitmap, [139, 87, 42])  # [240, 131, 176]
    obj_rough_trail = sly.ObjClass("rough trail", sly.Bitmap, [155, 155, 155])  # [175, 175, 98]
    obj_small_vegetation = sly.ObjClass(
        "small vegetation", sly.Bitmap, [209, 255, 158]
    )  # alt [173, 208, 69] [137,234,96]
    obj_forest = sly.ObjClass("forest", sly.Bitmap, [59, 93, 4])  # alt [3, 48, 0] [58, 155, 9]
    obj_sky = sly.ObjClass("sky", sly.Bitmap, [74, 144, 226])  # [21, 140 250]
    obj_obstacles = sly.ObjClass("obstacles", sly.Bitmap, [185, 20, 124])  # [255, 25, 0]

    rgbs = {
        (139, 87, 42): "smooth trail",
        (240, 131, 176): "smooth trail",
        (155, 155, 155): "rough trail",
        (175, 175, 98): "rough trail",
        (209, 255, 158): "small vegetation",
        (173, 208, 69): "small vegetation",
        (137, 234, 96): "small vegetation",
        (59, 93, 4): "forest",
        (3, 48, 0): "forest",
        (58, 155, 9): "forest",
        (74, 144, 226): "sky",
        (21, 140, 250): "sky",
        (185, 20, 124): "obstacles",
        (184, 20, 124): "obstacles",
        (255, 25, 0): "obstacles",
    }

    classes = {
        1: "smooth trail",
        2: "rough trail",
        3: "small vegetation",
        4: "forest",
        5: "sky",
        6: "obstacles",
    }
    labels2cls = {v: k for k, v in classes.items()}

    tag_metas = [sly.TagMeta(name, sly.TagValueType.NONE) for name in tag_names]

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[
            obj_smooth_trail,
            obj_rough_trail,
            obj_small_vegetation,
            obj_forest,
            obj_sky,
            obj_obstacles,
        ],
        tag_metas=tag_metas,
    )
    api.project.update_meta(project.id, meta.to_json())

    project_images = {}
    ds_names = []

    def list_all_paths(directory):
        paths = []
        for dirpath, _, filenames in os.walk(directory):
            for file in filenames:
                paths.append(os.path.join(dirpath, file))
        return paths

    all_paths = list_all_paths(dataset_path)
    all_dirpaths = list(set([os.path.dirname(path) for path in all_paths]))

    for dirpath in all_dirpaths:
        if "/raw_" in dirpath:
            if "Dataset4_NorthSlope_Segmentation/Dataset1/" in dirpath:
                continue
            else:
                ds_name = dirpath.split("/")[-2]
        elif "/imgs" in dirpath:
            if "/Train" in dirpath:
                ds_name = "Train"
            elif "/Test" in dirpath and "mixed" not in dirpath:
                ds_name = "Test"
        else:
            continue
        filepaths = [path for path in all_paths if dirpath in path]

        if "/Test" in dirpath:
            if project_images.get(ds_name) is None:
                project_images[ds_name] = filepaths
            else:
                project_images[ds_name] = project_images[ds_name] + filepaths
        else:
            project_images[ds_name] = filepaths

        ds_names.append(ds_name)

    def clean_string(s):
        s = re.sub(r"_{2,}", "_", s)
        s = re.sub(r"^_|_$", "", s)
        return s

    for ds_name in list(set(ds_names)):
        img_paths = project_images[ds_name]

        if ds_name == "Dataset1B - Powerline":
            ds_name = "dataset1B_powerline"

        elif ds_name == "Main_Trail":
            ds_name = "dataset1C_main_trail"

        elif ds_name == "Dataset1A-Brown_field":
            ds_name = "dataset1A_brown_field"

        elif ds_name == "Dataset2":
            ds_name = "dataset4_north_slope"
        else:
            ds_name = ds_name.replace("-", "_")
            ds_name = camel_to_snake(ds_name).lower()
            ds_name = clean_string(ds_name.split("_segmentation")[0])

        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)
        progress = sly.Progress("Create dataset '{}'".format(ds_name), len(img_paths))

        for img_pathes_batch in sly.batched(img_paths, batch_size=batch_size):
            img_names_batch = [os.path.basename(img_path) for img_path in img_pathes_batch]
            # if ds_name == "test":
            #     img_names_batch = [
            #         f"{os.path.basename(os.path.dirname(os.path.dirname(img_path))).lower()}_{os.path.basename(img_path)}"
            #         for img_path in img_pathes_batch
            #     ]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)

            img_ids = [im_info.id for im_info in img_infos]
            anns_batch = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(img_names_batch))

    return project
