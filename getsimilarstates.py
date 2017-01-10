
import sys
from sklearn.metrics.pairwise import cosine_similarity		
import pickle

"""
We are talking data as a simple csv and i'm reshaping the data from long to wide format
So for a given state getting a DictVectorizer that holds the freq of indivisual
male/female in columns

This will essentially look like 

statename |Ashley_F |Anna_F|. . . . 			 
	AK	  |36		| 34   |. . . . 
	.
	.		

We then use these 22 different vectors to calculate similarities b\w them

"""
def readdata(datafile,header=True):
    data=[]
    state_dict={}
    name_dict={}
    f=open(datafile,'r')
    
    for line in f:
        if header:
            header=False
            continue
        parts=line.strip().split(',')
        #print parts
        statename=parts[0]
        gender=parts[1]
        name=parts[2]
        freq=int(parts[3])
        
        if statename in state_dict:
            name_dict[name+'_'+gender]=freq
            state_dict[statename]=name_dict
        else:
            if name+'_'+gender in name_dict:
                name_dict[name+'_'+gender]=+freq
            else:
                name_dict[name+'_'+gender]=freq
            state_dict[statename]=name_dict
            name_dict={}
        
        
    return state_dict


"""
Given the data file will call readdata and calculate the 
pairwise cosine similarities using sklearn.

http://scikit-learn.org/stable/modules/metrics.html#metrics

We then serialize the similarities dictionary and because we are
using index and the name of states interchangibly we need to create
mapping and serialize that too.

Once we are done with train process we just call get_most_sim(test_statename)
and that will praduce most similar states

We can also explore more similarities 

"""
def calculate_sim(datafile):
	X=readdata(datafile)
	D=[]
	states_name=[]
	states_id_name={}
	states_name_id={}
	for state,val in X.items():
		states_name.append(state)
		D.append(val)
	for i,st in enumerate(states_name):
		states_id_name[i]=st
		states_name_id[st]=i


	from sklearn.feature_extraction import DictVectorizer
	v = DictVectorizer(sparse=False)

	X = v.fit_transform(D)
	similarities = cosine_similarity(X)
	state_sim_dict={}
	no_of_similar_states=3

	for i in range(0,len(similarities)-1):
		pairwise_sim=[]
		for j in range(0,len(similarities)-1):
			pairwise_sim.append((j,similarities[i][j]))
		for idx,s in sorted(pairwise_sim,key=lambda x: x[1],reverse=True):
			if idx==i:
				continue
			if i in state_sim_dict:
				state_sim_dict[i].append(idx)
			else:
				state_sim_dict[i]=[idx]

	
	states_id_name_file = 'states_id_name.pkl'
	states_name_id_file='states_name_id.pkl'
	state_sim_dict_file='state_sim_dict.pkl'

	pickle.dump(states_id_name, open(states_id_name_file, 'wb'))
	pickle.dump(states_name_id, open(states_name_id_file, 'wb'))
	pickle.dump(state_sim_dict, open(state_sim_dict_file, 'wb'))
	
	return state_sim_dict


"""
Given the test_statename map it to inedex using mapping, retrive the 
similar states using state_sim_dict and then show topk using reverse mapping.

"""

def get_most_sim(test_statename):
	topk=3
	states_id_name = pickle.load(open('states_id_name.pkl', 'rb'))
	states_name_id = pickle.load(open('states_name_id.pkl', 'rb'))
	state_sim_dict = pickle.load(open('state_sim_dict.pkl', 'rb'))
	print test_statename
	#print states_name_id
	#print states_id_name
	#print state_sim_dict
	idx=states_name_id[test_statename]
	#print idx
	gettopsim_states=state_sim_dict[idx]

	print gettopsim_states
	for st in gettopsim_states[0:topk]:
		print states_id_name[st]


def main():
	if len(sys.argv)<2:
		print "Please provide more arguments "
		print "for traing use <getsimilarstates.py><train><datafile>"
		print "for testing use <getsimilarstates.py><test><statename such as 'DE'>"
	elif sys.argv[1]=='train':
		calculate_sim(sys.argv[2])
	elif sys.argv[1]=='test':
		get_most_sim(sys.argv[2])
if __name__ == '__main__':
	main()