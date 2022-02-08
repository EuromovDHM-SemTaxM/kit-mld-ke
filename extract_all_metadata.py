#!/usr/bin/python
import json
import re
import traceback
from pathlib import Path

import tqdm as tqdm

import Glacier2, Ice, sys


# properties = Ice.createProperties(sys.argv)
# properties.load('client.cfg')
# init_data = Ice.InitializationData()
# init_data.properties = properties

def extract_description_tree(description):
    description_record = {'id': description.id, 'label': description.label}
    if len(description.children) == 0:
        description_record['children'] = []
    else:
        description_record['children'] = [extract_description_tree(child) for child in description.children]
    return description_record


def extract_cmu_suject_categorized_trials(url, cat, subcat, subject_trial_index):
    import requests
    from bs4 import BeautifulSoup

    page_content = requests.get(url)
    root = BeautifulSoup(page_content.text, features="html.parser")
    root = root.find('table')
    root = root.find_all('table')[-1]

    rows = root.find_all('tr')
    current_subject = -1
    for row in rows:
        if len(row.text.strip()) != 0:
            tds = row.find_all('td')
            title = tds[0].text
            content = tds[1].text
            if "Motion Description" in content:
                result = re.match("Subject\s([0-9]+) - Motion Description", content)
                current_subject = int(result.group(1))
                if current_subject not in subject_trial_index:
                    subject_trial_index[current_subject] = {}
            if re.match("[1-9]+", title):
                subject_trial_index[current_subject][int(title)] = \
                    {'trial': int(title), 'description': content, 'category': cat, 'sub_category': subcat}

    return subject_trial_index

def extract_cmu_all_trials(subject_trial_index):
    import requests
    from bs4 import BeautifulSoup

    page_content = requests.get("http://mocap.cs.cmu.edu/search.php")
    root = BeautifulSoup(page_content.text, features="html.parser")
    root = root.find('table')
    root = root.find_all('table')[-1]

    rows = root.find_all('tr')
    current_subject = -1
    for row in rows:
        if len(row.text.strip()) != 0:
            tds = row.find_all('td')
            title = tds[1].text
            content = tds[2].text
            if "Subject #" in tds[0].text:
                result = re.match("Subject\s#([0-9]+).*", tds[0].text)
                current_subject = int(result.group(1))
                if current_subject not in subject_trial_index:
                    subject_trial_index[current_subject] = {}
            if re.match("[1-9]+", title):
                if int(title) not in subject_trial_index[current_subject]:
                    subject_trial_index[current_subject][int(title)] = \
                        {'trial': int(title), 'description': content}

    return subject_trial_index


def extract_cmu_taxonomy():
    import requests
    from bs4 import BeautifulSoup

    taxonomy = []
    motion_registry = {}

    root_page = "http://mocap.cs.cmu.edu/motcat.php"
    tax_content = requests.get(root_page)
    tax_root = BeautifulSoup(tax_content.text, features="html.parser")
    tax_root = tax_root.find('table')
    tax_root = tax_root.find_all('table')[-1]
    root_links = tax_root.find_all('a')
    for link in root_links:
        href = link['href']
        descr = link.text
        id = int(href.split('?')[1].split('=')[1])
        children = []

        sub_cat_content = requests.get("http://mocap.cs.cmu.edu/" + href)
        sub_cat_root = BeautifulSoup(sub_cat_content.text, features="html.parser")

        sub_cat_root = sub_cat_root.find('table')
        sub_cat_root = sub_cat_root.find_all('table')[-1]
        subcat_links = sub_cat_root.find_all('a')
        for slink in subcat_links:
            shref = slink['href']
            sdescr = slink.text
            sid = int(shref.split('&')[1].split('=')[1])
            children.append({'id': sid, 'description': sdescr})
            motion_registry = extract_cmu_suject_categorized_trials("http://mocap.cs.cmu.edu/" + shref, id, sid, motion_registry)

        taxonomy.append({'id': id, 'name': descr, 'children': children})

    return taxonomy, motion_registry


kit_metadata = {}
cmu_metadata = {}
non_kit_metadata = {}
datapath = Path("data/")
for file in datapath.glob("*_meta.json"):
    id = int(str(file.stem).split("_")[0])
    with open(file, "r") as m:
        meta = json.load(m)
        if meta['source']['database']['identifier'] == 'kit':
            meta['mld_id'] = id
            kit_metadata[meta['source']['database']['motion_id']] = meta
        elif meta['source']['database']['identifier'] == 'cmu':
            meta['mld_id'] = id
            cmu_metadata[meta['source']['database']['motion_id']] = meta
        else:
            non_kit_metadata[id] = meta

final_data = {}

taxonomy, motion_registry = extract_cmu_taxonomy()
motion_registry = extract_cmu_all_trials(motion_registry)
final_data['cmu_description_taxonomy'] = taxonomy

for kit_id in cmu_metadata:
    motion_metadata = cmu_metadata[kit_id]
    subject = int(meta['source']['database']['motion_id'])
    trial = int(meta['source']['database']['motion_file_id'])
    record = {}
    record['metadata'] = motion_metadata
    record['id'] = motion_metadata['mld_id']
    record.update(motion_registry[subject][trial])
    final_data[motion_metadata['mld_id']] = record

with Ice.initialize(sys.argv, 'client.cfg') as ic:
    Ice.loadSlice(f"-I{Ice.getSliceDir()}  MotionDatabase.ice")
    import MotionDatabase

    router = Glacier2.RouterPrx.checkedCast(ic.getDefaultRouter())
    session = router.createSession('', '')  # Username and password (leave empty for anonymous access)
    db = MotionDatabase.MotionDatabaseSessionPrx.checkedCast(session)
    print(f'Connected to Glacier2 router: {db}')

    count = db.countMotions(None, None, None, None, None, None)
    print(count)
    batch_size = 200
    motions = {}

    motion_description_tree = db.getMotionDescriptionTree()
    motion_tree_dicts = [extract_description_tree(root) for root in motion_description_tree]

    final_data['kit_description_taxonomy'] = motion_tree_dicts
    for offset in tqdm.trange(0, count, batch_size):
        for motion in db.listMotions(None, None, None, None, None, None, "id", batch_size, offset):
            if motion.id in kit_metadata:
                record = {}
                motion_metadata = kit_metadata[motion.id]
                record['metadata'] = motion_metadata
                record['id'] = motion_metadata['mld_id']
                record['kit_motion_id'] = motion.id

                record['descriptions'] = [extract_description_tree(description) for description in
                                          motion.motionDescriptions]

                institution = motion.associatedInstitution
                record['institution_id'] = institution.id
                record['institution_acronym'] = institution.acronym
                record['institution_name'] = institution.name

                record['project_name'] = motion.associatedProject.name

                subjects = motion.associatedSubjects
                record['subjects'] = [
                    {'firstname': subject.firstName, 'lastname': subject.lastName, 'comment': subject.comment,
                     'gender': subject.gender, 'age': subject.age, 'weight': subject.weight,
                     'height': subject.height, 'antropomorphology': subject.anthropometricsTable} for subject in
                    subjects]
                objects = motion.associatedObjects
                record['objects'] = \
                    [
                        {'label': object.label, 'comment': object.comment, 'model_setting': object.modelSettingJSON}
                        for object in objects
                    ]
                record['date'] = motion.date
                record['comment'] = motion.comment
                final_data[motion_metadata['mld_id']] = record

for kit_id in non_kit_metadata:
    final_data[kit_id] = {'metadata': non_kit_metadata[kit_id]}

with open("motion_data_full.json", "w") as fp:
    json.dump(final_data, fp, indent=4)
