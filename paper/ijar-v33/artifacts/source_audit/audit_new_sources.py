"""Source-only audit. Does not load CIFAR-10.1 labels or any model output."""
from pathlib import Path
import argparse,csv,json,hashlib,pickle
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--new-data',required=True);a.add_argument('--reference',required=True);a.add_argument('--out',required=True);q=a.parse_args()
 src=Path(q.new_data);ref=Path(q.reference);out=Path(q.out);out.mkdir(parents=True,exist_ok=True)
 if (out/'SOURCE_AUDIT.json').exists():raise RuntimeError('Refuse overwrite of source audit')
 images=np.load(src/'cifar10.1_v6_data.npy',allow_pickle=False)
 ids=np.asarray(json.loads((src/'cifar10.1_v6_ti_indices.json').read_text()),dtype=np.int64)
 assert images.shape==(2000,32,32,3) and images.dtype==np.uint8
 assert len(ids)==len(images) and len(set(map(int,ids)))==2000
 refhash={};rf=[]
 for name in ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']:
  p=ref/name;d=pickle.loads(p.read_bytes(),encoding='bytes')
  x=d[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1)
  rf.append({'file':str(p),'sha256':sha(p),'rows':len(x)})
  for i,img in enumerate(x):refhash.setdefault(hashlib.sha256(img.tobytes()).hexdigest(),[]).append(f'{name}:{i}')
  del d,x
 records=[];seen={};retained=[]
 for row in np.argsort(ids):
  row=int(row);ti=int(ids[row]);h=hashlib.sha256(images[row].tobytes()).hexdigest();reason=None
  if h in refhash:reason='EXACT_PIXEL_OVERLAP_CIFAR10_REFERENCE'
  elif h in seen:reason='WITHIN_NEW_FRAME_PIXEL_DUPLICATE'
  else:seen[h]=ti;retained.append(row)
  records.append({'row_index':row,'source_id':f'tinyimages:{ti}','tinyimages_id':ti,'pixel_sha256':h,'retained':reason is None,'exclusion':reason,'reference_matches':refhash.get(h,[])})
 retained=sorted(retained)
 ranked=sorted(retained,key=lambda row:(hashlib.sha256(f'fedcore-independent-20260914:{int(ids[row])}'.encode()).hexdigest(),int(ids[row])))
 client={row:k%5 for k,row in enumerate(ranked)}
 for r in records:r['client']=client.get(r['row_index'])
 records.sort(key=lambda x:x['row_index'])
 (out/'SOURCE_LEDGER.json').write_text(json.dumps(records,indent=2))
 np.savez_compressed(out/'public_source_frame.npz',row_index=np.array(retained),source_ids=np.array([f'tinyimages:{int(ids[i])}' for i in retained]),pixel_sha256=np.array([records[i]['pixel_sha256'] for i in retained]),clients=np.array([client[i] for i in retained],dtype=np.int64))
 result={'status':'PASS','new_rows':len(images),'retained_rows':len(retained),'unique_upstream_source_ids':len(set(map(int,ids))),'exact_reference_overlaps':sum(r['exclusion']=='EXACT_PIXEL_OVERLAP_CIFAR10_REFERENCE' for r in records),'within_new_pixel_duplicates':sum(r['exclusion']=='WITHIN_NEW_FRAME_PIXEL_DUPLICATE' for r in records),'reference_rows':sum(x['rows'] for x in rf),'reference_files':rf,'new_images_sha256':sha(src/'cifar10.1_v6_data.npy'),'upstream_ids_sha256':sha(src/'cifar10.1_v6_ti_indices.json'),'source_ledger_sha256':sha(out/'SOURCE_LEDGER.json'),'public_frame_sha256':sha(out/'public_source_frame.npz'),'client_sizes':[sum(v==j for v in client.values()) for j in range(5)],'fresh_label_array_loaded':False,'model_outputs_loaded':False,'scope':'Unique upstream Tiny Images IDs and exact decoded-pixel hashes. This check alone does not rule out nonidentical near-duplicates; upstream dataset construction also performed near-duplicate filtering.'}
 (out/'SOURCE_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='reference_files'},indent=2))
if __name__=='__main__':main()
