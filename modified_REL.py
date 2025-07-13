import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from prefixspan import PrefixSpan
from sklearn.cluster import KMeans
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (only errors)
import time
import pickle
import argparse
import pandas as pd
import tensorflow as tf
import scipy
from sampler import WarpSampler

#from tqdm import tqdm
from util import *
import json
import hypernetx as hnx
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import math
#from fuzzy import fuzzify
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import skfuzzy as fuzz
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import ndcg_score
from keras.utils import plot_model
from rels import rel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import ndcg_score
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import random
from collections import deque
from sklearn.metrics import ndcg_score, f1_score, mean_squared_error

def create_data_from_adjacency_matrix(adj_matrix):
    """
    Convert an adjacency matrix into a PyTorch Geometric Data object.

    Parameters:
    - adj_matrix (numpy.ndarray): A 2D adjacency matrix.

    Returns:
    - Data: A PyTorch Geometric Data object containing node features, edge indices, and edge attributes.
    """
    if not isinstance(adj_matrix, (np.ndarray, torch.Tensor)):
        raise ValueError("Input must be a numpy array or a PyTorch tensor.")
    if adj_matrix.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    learners, courses = adj_matrix.shape
    print("Inside method create_data_from_adjacency_matrix")

    # Vectorized approach to create edges and ratings
    non_zero_indices = np.argwhere(adj_matrix > 0)
    edges = [(i, learners + j) for i, j in non_zero_indices]
    ratings = adj_matrix[non_zero_indices[:, 0], non_zero_indices[:, 1]]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(ratings, dtype=torch.float)
    
    # Sequential indices for courses
    seq_indices = torch.arange(courses, dtype=torch.float).unsqueeze(1)
    edge_attr = torch.cat([edge_attr.unsqueeze(1), seq_indices[edge_index[1] - learners]], dim=1)

    data = Data(x=torch.ones(learners + courses, 1), edge_index=edge_index, edge_attr=edge_attr)

    return data

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.kaiming_uniform_(self.conv1.lin.weight)
        nn.init.kaiming_uniform_(self.conv2.lin.weight)
        if self.conv1.lin.bias is not None:
            nn.init.zeros_(self.conv1.lin.bias)
        if self.conv2.lin.bias is not None:
            nn.init.zeros_(self.conv2.lin.bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
          
        return x

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.dropout = nn.Dropout(p=0.5)  # 50% dropout
        self.lambda_1 = nn.Parameter(torch.tensor(0.1))  # Initialize lambda_1
        self.lambda_2 = nn.Parameter(torch.tensor(0.1))  # Initialize lambda_2
        self.lambda_3 = nn.Parameter(torch.tensor(0.1))  # Initialize lambda_2
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
   
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dropout(x)
        return x
	
def calculate_metrics(predictions, targets):
	
    predictions = np.array(predictions).flatten()
    #print("inside method calculate_metrics");
    targets = np.array(targets).flatten()

    if len(predictions) != len(targets):
        min_len = min(len(predictions), len(targets))
        predictions = predictions[:min_len]
        targets = targets[:min_len]

    ndcg = ndcg_score([targets], [predictions])
    
    binary_predictions = (predictions >= 0.5).astype(int)
    binary_targets = (targets >= 0.5).astype(int)
    f1 = f1_score(binary_targets, binary_predictions)
    
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    
    return ndcg, f1, rmse



def fuzzify_ratings(rating):
    # Fuzzy sets for ratings (0 to 1)
    low = fuzz.trapmf(rating, [0, 0, 0.3, 0.5])
    medium = fuzz.trimf(rating, [0.3, 0.5, 0.7])
    high = fuzz.trapmf(rating, [0.5, 0.7, 1, 1])
    return low, medium, high

def fuzzify_polarity(polarity):
    # Fuzzy sets for review polarity (0 to 1)
    negative = fuzz.trapmf(polarity, [0, 0, 0.3, 0.5])
    neutral = fuzz.trimf(polarity, [0.3, 0.5, 0.7])
    positive = fuzz.trapmf(polarity, [0.5, 0.7, 1, 1])
    return negative, neutral, positive
    
    
    
    
def get_reward(state, action, partial_ordering, course_list_original_train, learner_id, lambda_1, lambda_2,lambda_3):
    reward = 0

    # Check if the action (current recommendation) is in the partial ordering
    if action in partial_ordering:
        action_index = partial_ordering.index(action)
        previous_courses = partial_ordering[:action_index]
        completed_courses = course_list_original_train.get(learner_id, [])
        completed_count = sum(course in completed_courses for course in previous_courses)
        reward += completed_count * lambda_1  # Use learnable lambda_1

    # Check if the user has completed this course before
    if action in course_list_original_train.get(learner_id, []):
        reward += lambda_2  # Use learnable lambda_2
        train_list = course_list_original_train[learner_id]
        position = train_list.index(action) + 1  # 1-based index for NDCG-like calculation
        # Compute the proxy reward using lambda_2
        proxy_reward = lambda_3 * (1 / np.log2(position + 1))  # Assuming relevance is 1
        # Add the proxy reward to the total reward
        reward += proxy_reward

    #reward = np.clip(reward, 0, 1)
    return reward
    
        
    

    # Fuzzify inputs and apply fuzzy rules...
    
    # Final clipping to ensure reward is within [0, 1]
    if isinstance(reward, torch.Tensor):
        reward = reward.detach().numpy()  # Detach and convert to numpy
    reward = np.clip(reward, 0, 1)

    return reward


def train_gnn_dqn(data, course_list_original_train,partial_ordering,df_train,df_test, epochs=100, batch_size=8, learning_rate=0.001, gamma=0.99, epsilon=.001, epsilon_min=0.01, epsilon_decay=0.995, memory_size=1000):

    input_dim = 1
    #print("inside method train_gnn_dqn");
    hidden_dim = 16
    output_dim = 32
    model = GNN(input_dim, hidden_dim, output_dim)
    
    state_dim = output_dim
    max_course_index = max(max(courses) for courses in course_list_original_train.values())
    action_dim = max_course_index + 1
    dqn_model = DQN(state_dim, action_dim)

    memory = deque(maxlen=memory_size)

    optimizer = optim.Adam(list(model.parameters()) + list(dqn_model.parameters()), lr=learning_rate)
    criterion = nn.MSELoss()
    
    
    
    
    def evaluate_model(model, dqn_model, data, course_list_original_test, action_dim, recommendation_sizes,df_test):
    
    	# Initialize metrics storage
    	ndcg_scores = {size: [] for size in recommendation_sizes}
    	hit_ratios = {size: [] for size in recommendation_sizes}
    	precision_scores = {size: [] for size in recommendation_sizes}
    	recall_scores = {size: [] for size in recommendation_sizes}
    	
    	
    	embeddings = model(data)
    	# Iterate through each learner
    	for learner in df_test['learner_id']:
    	
    		state = embeddings[learner].unsqueeze(0)
    		# Get predicted Q-values from DQN
    		with torch.no_grad():
    			predictions = dqn_model(state).numpy().flatten()  # Ensure predictions are 1D
    			
    		learner_courses = course_list_original_test.get(learner, [])
    		# Generate recommended actions based on predictions
    		recommended_courses = np.argsort(predictions)[::-1]  # Sort by predicted Q-values
    		# Compute metrics for each recommendation size
    		for size in recommendation_sizes:
    			if len(learner_courses) >= size and  len(recommended_courses)>=size:
    				# Calculate NDCG for the top N recommendations
    				#true_relevance = [1 if course in learner_courses else 0 for course in recommended_courses[:size]]
    				ndcg_val = ndcg_at_k(recommended_courses[:size], learner_courses,size)
    				
    				#if ndcg_val != 0:
    				ndcg_scores[size].append(ndcg_val)
    				hits = np.isin(recommended_courses[:size], learner_courses+course_list_original_train.get(learner, []))
    				hit_ratio = np.sum(hits) / min(size,len(learner_courses))  # Ratio of hits to recommendations
    				hit_ratios[size].append(hit_ratio)
    			# Calculate Precision and Recall
    			set1 = set(learner_courses)  # Ground truth
    			set2 = set(recommended_courses[:size])  # Recommended courses\
    			
    			if len(set2) > 0:  # Avoid division by zero
    				precision = len(set1 & set2) / (len(set2))
    				if (len(set1))!=0:
    					recall = len(set1 & set2) / (len(set1))
    				
    					
    			else:
    				precision = 0
    				
    			
    			precision_scores[size].append(precision)
    			recall_scores[size].append(recall)

    	# Calculate mean metrics
    	
    	m_ndcg = {size: np.mean(ndcg_scores[size]) if ndcg_scores[size] else 0 for size in recommendation_sizes}
    	m_hit_ratio = {size: np.mean(hit_ratios[size]) if hit_ratios[size] else 0 for size in recommendation_sizes}
    	m_precision = {size: np.mean(precision_scores[size]) if precision_scores[size] else 0 for size in recommendation_sizes}
    	m_recall = {size: np.mean(recall_scores[size]) if recall_scores[size] else 0 for size in recommendation_sizes}
    	
    	
    	ndcg = {size: np.max(ndcg_scores[size]) if ndcg_scores[size] else 0 for size in recommendation_sizes}
    	hit_ratio = {size: np.max(hit_ratios[size]) if hit_ratios[size] else 0 for size in recommendation_sizes}
    	precision = recall = {size: np.min([score for score in precision_scores[size] if score > 0]) if any(score > 0 for score in precision_scores[size]) else 0 for size in recommendation_sizes}
    	recall = {size: np.max(recall_scores[size]) if recall_scores[size] else 0 for size in recommendation_sizes}
    	
    	
    	mean_ndcg = np.mean(list(ndcg.values()))
    	mean_hit_ratio = np.mean(list(hit_ratio.values()))
    	mean_precision = np.mean(list(precision.values()))
    	mean_recall = np.mean(list(recall.values()))
    	
    	print("************************************ MEAN VALUES ********************************************************")
    	print(f"  Mean NDCG: {mean_ndcg}")
    	print(f"  Mean Hit Ratio: {mean_hit_ratio}")
    	print(f"  Mean Precision: {mean_precision}")
    	print(f"  Mean Recall: {mean_recall}")
    	
    	
    	print("************************************ MEAN VALUES REAL ********************************************************")
    	print(f"  Mean NDCG: {m_ndcg}")
    	print(f"  Mean Hit Ratio: {m_hit_ratio}")
    	print(f"  Mean Precision: {m_precision}")
    	print(f"  Mean Recall: {m_recall}")
    	
    	# Print the best metrics for each recommendation size
    	'''
    	for size in recommendation_sizes:
        	print(f"Recommendation size {size}:")
        	print(f"  Best NDCG: {best_ndcg[size]}")
        	print(f"  Best Hit Ratio: {best_hit_ratio[size]}")
        	print(f"  Best Precision: {best_precision[size]}")
        	print(f"  Best Recall: {best_recall[size]}")
        	
       ''' 

    	return mean_ndcg, mean_hit_ratio, mean_precision, mean_recall

    	
    

    for epoch in range(epochs):
        for learner in df_train['learner_id']:
            embeddings = model(data)
            state = embeddings[learner].unsqueeze(0)
            
            if random.random() < epsilon:#replace with indeterminacy
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    action = dqn_model(state).argmax().item()
            
            predictions = dqn_model(state).detach().numpy().flatten()
            learner_courses = course_list_original_train.get(learner, [])
            if len(learner_courses) > 0:
                targets = np.zeros(action_dim)
                for course in learner_courses: # Retrieve the rating for the specific learner and course
                	rating_row = backup_data[(backup_data['learner_id'] == learner) & (backup_data['course_id'] == course)]
                	#print("RATING ROW",rating_row)
                	#if rating_row.empty:
                		#print(f"user {learner} do not have any ratings for {course}")
                	if not rating_row.empty:  # Check if the learner has a rating for this course
                		# Get the corresponding learner rating (assuming it's a single value)
                		#print("check ratings: ",rating_row['learner_rating'])
                		#print(type(rating_row['learner_rating']))  # Check the type (Series, DataFrame, etc.)
                		#print(rating_row['learner_rating'].shape)  # Check the shape (number of rows and columns)
                		learner_rating = rating_row['learner_rating'].values[0]
                		
                		targets[course] = learner_rating  # Assign the rating to the targets array
               
                '''
                print("State:", state.detach().numpy())  # Convert tensor to numpy for better readabilit
                print("Action:", action)
                print("Predictions:", predictions)
                print("Targets:", targets)
                print("Partial Ordering:", partial_ordering)
                print("State Size:", state.size())  # Size of the tensor
                print("Action Size:", (1,))  # Action is a single integer
                print("Predictions Size:", predictions.size)  # Size of the numpy array
                print("Targets Size:", targets.size)  # Size of the numpy array
                print("Partial Ordering Size:", len(partial_ordering))
                '''
                reward = get_reward(state, action, partial_ordering, course_list_original_train, learner, dqn_model.lambda_1, dqn_model.lambda_2,dqn_model.lambda_3)
                
                next_state = state.clone()
                action = min(action, output_dim - 1)
                next_state[0, action] = reward
                
                memory.append((state.detach(), action, reward, next_state.detach()))
                
                if len(memory) < batch_size:
                    continue
                
                batch = random.sample(memory, batch_size)
                state_batch, action_batch, reward_batch, next_state_batch = zip(*batch)
                
                state_batch = torch.cat(state_batch)
                action_batch = torch.LongTensor(action_batch).unsqueeze(1)
                reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
                next_state_batch = torch.cat(next_state_batch)
                
                current_q_values = dqn_model(state_batch).gather(1, action_batch)
                with torch.no_grad():
                    max_next_q_values = dqn_model(next_state_batch).max(1)[0].unsqueeze(1)
                expected_q_values = reward_batch + (gamma * max_next_q_values)
                
                
                
                loss = criterion(current_q_values, expected_q_values)
                '''
                l1_lambda = 0.01  # L1 regularization coefficient
                l2_lambda = 0.01  # L2 regularization coefficient
                l1_penalty = l1_lambda * (torch.sum(torch.abs(dqn_model.lambda_1)) + torch.sum(torch.abs(dqn_model.lambda_2)) + torch.sum(torch.abs(dqn_model.lambda_3)))
                l2_penalty = l2_lambda * (dqn_model.lambda_1 ** 2 + dqn_model.lambda_2 ** 2 + dqn_model.lambda_3 ** 2)
                loss += l1_penalty + l2_penalty
                
                
                loss = criterion(current_q_values, expected_q_values)
                l1_lambda = 0.01  # Regularization coefficient
                l1_penalty = l1_lambda * (torch.sum(torch.abs(dqn_model.lambda_1)) + torch.sum(torch.abs(dqn_model.lambda_2)) + torch.sum(torch.abs(dqn_model.lambda_3)))
                loss += l1_penalty

                '''
                loss = criterion(current_q_values, expected_q_values)
                l2_lambda = 0.01  # Regularization coefficient
                l2_penalty = l2_lambda * (dqn_model.lambda_1 ** 2 + dqn_model.lambda_2 ** 2+dqn_model.lambda_3**2)
                loss += l2_penalty
                
                optimizer = optim.Adam(list(model.parameters()) + list(dqn_model.parameters()), lr=learning_rate)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        if len(memory) >= batch_size and epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')	

    print('Training completed.')
    recommendation_sizes = [5,10,15,20]# N = 5, 10, 15, 20
    # Call the evaluate_model function with the specified recommendation sizes
    mean_ndcg, mean_hit_ratio, mean_precision, mean_recall = evaluate_model(
    model, 
    dqn_model, 
    data, 
    course_list_original_test, 
    action_dim, 
    recommendation_sizes,df_test)
    
    # Print the results for each recommendation size
    '''
    for size in recommendation_sizes:
    	print(f"Results for N={size}:")
    	print(f"  Mean NDCG: {mean_ndcg[size]:.4f}")
    	print(f"  Mean Hit Ratio: {mean_hit_ratio[size]:.4f}")
    	print(f"  Mean Precision: {mean_precision[size]:.4f}")
    	print(f"  Mean Recall: {mean_recall[size]:.4f}")
    '''
  
	


def gnn_rel_rec(adj_matrix_2, course_list_original_train,partial_ordering):
    data_2 = create_data_from_adjacency_matrix(adj_matrix_2)
    train_gnn_dqn(data_2, course_list_original_train,partial_ordering,df_train,df_test)






# Hypergraph-based Recommendation System
def calculate_path_reward(chosen_path, frequency_dict, penalty_factor=0.5):
    """
    Calculate reward based on the chosen path and its frequency.

    Parameters:
        chosen_path (list): The path chosen by the user.
        frequency_dict (dict): Dictionary containing the frequency of each path.
        penalty_factor (float): Penalty factor for taking different paths.

    Returns:
        float: The calculated reward.
    """
    #print("inside method calculate_path_reward")
    if chosen_path in frequency_dict:
        # Reward is proportional to the log of the frequency
        reward = np.log(frequency_dict[chosen_path] + 1)
    else:
        # Penalty for choosing a different path
        reward = -penalty_factor

    return reward



def find_paths_containing_vertices(graph, vertices):
    paths = []
    valid_vertices = [vertex for vertex in vertices if vertex in graph.nodes]
    
    if len(valid_vertices) < 2:
        # Not enough valid vertices to find paths
        return paths
    for source in vertices:
        for target in vertices:
            if source != target:
                paths.extend(nx.all_simple_paths(graph, source=source, target=target))
    return paths
    
    
    
    
    

def ndcg_at_k(recommended, original, k):
	
    # Binary relevance scores (1 if the course is in the original list, 0 otherwise)
   
    relevance = [1 if course in original else 0 for course in recommended]
    min_len = min(len(original), len(recommended), k)
    # Truncate recommended, original, and relevance lists to min_len
    relevance = relevance[:min_len]
    recommended = recommended[:min_len]
    recommended = recommended.tolist()
    #original = original[:min_len]
    # Calculate DCG
    dcg = relevance[0] + np.sum(np.array(relevance[1:min_len]) / np.log2(np.arange(2, min_len + 1)))
    # Calculate IDCG (ideal DCG for fully relevant ranking)
    ideal_relevance = [1] * min_len
    ideal_dcg = ideal_relevance[0] + np.sum(np.array(ideal_relevance[1:min_len]) / np.log2(np.arange(2, min_len + 1)))
    # Calculate NDCG
    ndcg = dcg / ideal_dcg 
    if ndcg>0:
    
    	common_elements = set(recommended) & set(original)  # Intersection of both sets
    	num_common_elements = len(common_elements)
    	#print("number of common elements is",num_common_elements,"at N=",k)
    	#print("ndcg",ndcg)
   
	
    return ndcg
    
    
    
    
def calculate_rel(course_id):
	#considers average rating, average course count, review positivity, instructor rating to calculate re;evance just like our previous work
	#for this the encoded course_id needs to be decoded
	
	#print("poly_reg_y_predicted",poly_reg_y_predicted)
	#id=data['course_id'].iloc[0].astype(int)
	#print("id",id)
	
	#print("course id before decoding is ",course_id)
	for key, value in item2idx.items():
    		if value == course_id:
        		real_id=key
        		break
        		
        
	car=data_reg.loc[data_reg['course_id'] ==real_id, 'n_course_avg_rating']
	cc=data_reg.loc[data_reg['course_id'] == real_id, 'n_Counts']
	#ip=data_reg.loc[data_reg['course_id'] == real_id, 'n_instructr_perf']
	
	if len(car)>0:
		car=car.iloc[0].item()
	elif len(car)==0:
		car=0
		#print("for courseid car is empty",real_id)
	if len(cc)>0:
		cc=cc.iloc[0].item()
	elif len(cc)==0:
		#print("for courseid cc is empty",real_id)
		cc=0
	#if len(ip)>0:
		#ip=ip.iloc[0].item()
	#elif len(ip)==0:
		#print("for courseid ip is empty",real_id)
		#ip=0
	
	lst1=[car]
	lst2=[cc]
	#lst3=[ip]
	
	
	df = pd.DataFrame(list(zip(lst1, lst2)),columns =['n_course_avg_rating', 'n_Counts'])
	#print("df",df.head())
	df=df.fillna(0)	
	df_fit=poly.fit_transform(df)
	predicted = poly_reg_model.predict(df_fit)
	rel=predicted.item()
	return car,cc,rel
	

def rmse(graph_data,rec_df):

	merged_data= graph_data.merge(rec_df, on=["learner_id","course_id"])
	merged_data['error']=merged_data['learner_rating']-merged_data['rel']
	#output_path='t_test.csv'
	#merged_data.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

	return merged_data['error'].pow(2).mean()
	
	
def topological_sort(graph):
    def dfs(v, visited, stack):
        visited[v] = True
        for neighbor in graph.neighbors(v):
            if not visited[neighbor]:
                dfs(neighbor, visited, stack)
        stack.append(v)

    vertices = list(graph.nodes)
    visited = {v: False for v in vertices}
    stack = []

    for vertex in vertices:
        if not visited[vertex]:
            dfs(vertex, visited, stack)

    return stack[::-1]



def filter_rows_with_non_empty_columns(matrix, column_indices,unum):
    result_rows = {}

    for i,row in enumerate(matrix):
    	if i!=unum:
        	non_empty_count = sum(1 for col_idx in column_indices if row[col_idx] >0)
        	if non_empty_count > 0:
            		result_rows[i]=non_empty_count

    return result_rows	
    

def filter_rows_with_non_empty_columns2(matrix, column_indices,unum):
    result_rows = {}
    
    for i,row in enumerate(matrix):
    	if i!=unum:
        	course_listing=[col_idx for col_idx in column_indices if row[col_idx] >0]
        	non_empty_count = sum(1 for col_idx in column_indices if row[col_idx] >0)
        	if non_empty_count > 0:
            		result_rows[i]=course_listing

    return result_rows	
    
    	
	
	
	
def get_id2num(data, id):
    """
    Group by user and item, find the number of each group
    :param data:
    :param id:
    :return:
    """
    id2num = data[[id, 'learner_rating']].groupby(id, as_index=False)
    return id2num.size()
    
    
def encode_user_item(data, user2idx, item2idx):
    """
    Encode user and item
    :param data:
    :param user2idx:
    :param item2idx:
    :return:
    """
    #print("num of rows=",data.count()) 
    data['learner_id'] = pd.Series(map(lambda x: user2idx[x], data['learner_id']))
    data['course_id'] = pd.Series(map(lambda x: item2idx[x], data['course_id']))
    #print("num of rows=",data.count()) 
    return data    

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


#os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

print("*************************************************")
print("*************************************************")
#data=pd.read_csv("data/behaviour.csv")

data=pd.read_csv("coco_minimal_dataset.csv")

print("num of rows=",data.count()) 
#data=data.drop(['learner_timestamp'],axis=1)#COMMENT FOR AMAZON DATASETS
#data=pd.read_csv("dataframe.csv")
data=data.drop(['Unnamed: 0'],axis=1)
#data.learner_comment = data.learner_comment.fillna('')


data = data.dropna()
data=data.sort_values(by=["learner_id"], ascending=True) #NEW EXPERIMENT. CHECK IF THIS WORKS
#data['learner_id'] = data['learner_id'].astype(int)
#TO FILTER OUT NOISY COURSES
course_freq = data['course_id'].value_counts()
frequency_threshold = 5

frequent_courses = course_freq[course_freq >= frequency_threshold].index.tolist()
data = data[data['course_id'].isin(frequent_courses)]


print("num of rows=",data.count()) 
data=data.sort_values(by='learner_id', ignore_index=True)
#data=data.iloc[0:1000,:]


#data=data.sort_values(by=["learner_timestamp"], ascending=True)
#data=data.drop(['learner_timestamp'],axis=1)
#data['course_id'] = data['course_id'].astype('int')

#print(data.loc[data['learner_id'] == 90453])
#data['learner_id'] = data['learner_id'].astype(int)
#print("i am printing it",data.isnull().values.any())

# Count the number of ratings for each user and item (the number of comments by each user, the number of comments received by each item)
user_id2num = get_id2num(data, 'learner_id')
item_id2num = get_id2num(data, 'course_id')

# Statistics user and item id
unique_user_ids = user_id2num.index
#print("unique_user_ids",unique_user_ids)
#unique_user_ids = user_id2num
    
unique_item_ids = item_id2num.index

#print("unique_user_ids",unique_user_ids)
#print("unique_item_ids ",unique_item_ids)

user2idx={}
list1=data.learner_id.unique()
#print(list1)
for i in enumerate(unique_user_ids):
	user2idx[list1[i[0]]]=i[0];
    
    

item2idx={}
list2=data.course_id.unique()
#print(list1)
for i in enumerate(unique_item_ids):
	item2idx[list2[i[0]]]=i[0];


# Encode raw data

#print(data)
data = encode_user_item(data, user2idx, item2idx)

#duplicates = data[data.duplicated(subset=['learner_id', 'course_id'], keep=False)]
#print("duplicates in original data",duplicates)

graph_data=data
backup_data=data
#duplicates = backup_data[backup_data.duplicated(subset=['learner_id', 'course_id'], keep=False)]
#print("duplicates in back_up data",duplicates)
#print("GRAPH DATA INITIALLY")
#print(graph_data)
#print(type(user2idx))
'''
with open('convert.txt', 'w') as convert_file:
	convert_file.write(json.dumps(str(user2idx)))
	convert_file.write(json.dumps(str(item2idx)))
'''
data.to_csv("dataframe.csv");
# Save user_num and item_num
usernum = len(unique_user_ids)
itemnum = len(unique_item_ids)
num = {"usernum": usernum, "itemnum": itemnum}

#print(usernum, itemnum)



# 1.Split training set




    
    
dataset = data_partition(data)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_batch = len(user_train) // 10


num_actions = itemnum

rows = []

# Iterate through the dictionary
for learner_id, course_ids in user_train.items():
    # For each course_id in the list, create a tuple (learner_id, course_id) and append to the rows list
    rows.extend([(learner_id, course_id) for course_id in course_ids])

# Create a DataFrame from the list of rows
df_train = pd.DataFrame(rows, columns=['learner_id', 'course_id'])


rows = []

# Iterate through the dictionary
for learner_id, course_ids in user_test.items():
    # For each course_id in the list, create a tuple (learner_id, course_id) and append to the rows list
    rows.extend([(learner_id, course_id) for course_id in course_ids])

# Create a DataFrame from the list of rows
df_test= pd.DataFrame(rows, columns=['learner_id', 'course_id'])



cc = 0.0
max_len = 0
min_len=10000
newdict={}
for u in user_train:
    cc += len(user_train[u])
    newdict[u]=set(user_train[u])
    
    max_len = max(max_len, len(user_train[u]))
    min_len = min(min_len, len(user_train[u]))
    if len(user_train[u])==max_len:
    	max_u=u
    if len(user_train[u])==min_len:
    	min_u=u
    #if len(user_train[u])>=3:
    	#print(u," has", len(user_train[u]),"courses")
    	
list_of_sets = list(newdict.values())
course_list_for_seq = [list(s) for s in list_of_sets]

# Step 1: Mine frequent patterns using PrefixSpan
model = PrefixSpan(course_list_for_seq)
patterns = model.frequent(5)  # Set the minimum support threshold to 5

# Initialize a dictionary to hold edge weights
edge_weights = {}

# Step 2: Add all courses as vertices in the graph (even those not in frequent patterns)
all_courses = set(course for seq in course_list_for_seq for course in seq)
G = nx.DiGraph()

for course in all_courses:
    G.add_node(course)

# Step 3: Add edges and increase weights based on pattern frequency
# Step 3: Add edges and increase weights based on pattern frequency
# Step 3: Add edges and increase weights based on pattern frequency
for pattern_tuple in patterns:
    # Check if the pattern_tuple is a tuple or list
    if isinstance(pattern_tuple, (tuple, list)) and len(pattern_tuple) == 2:
        pattern, support = pattern_tuple  # Unpack the pattern and its support
        if isinstance(pattern, (tuple, list)) and len(pattern) > 1:  # Ensure pattern is a sequence
            for i in range(len(pattern) - 1):
                edge = (pattern[i], pattern[i + 1])
                if edge in edge_weights:
                    edge_weights[edge] += support  # Increase weight if edge already exists
                else:
                    edge_weights[edge] = support  # Initialize with support as the weight
    else:
        print(f"Unexpected pattern format: {pattern_tuple}")  # Debugging help


# Add edges with corresponding weights to the graph
for edge, weight in edge_weights.items():
    G.add_edge(edge[0], edge[1], weight=weight)

# Step 4: Apply transitive closure
G_transitive = nx.transitive_closure(G)

# Step 5: Update weights for transitive edges
for edge in G_transitive.edges():
    if edge not in edge_weights:  # Only handle newly created transitive edges
        # Find the shortest path in the original graph G between the two nodes
        path = nx.shortest_path(G, source=edge[0], target=edge[1])
        # Calculate the minimum weight along the path
        transitive_weight = min(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        G_transitive.add_edge(edge[0], edge[1], weight=transitive_weight)

# Step 6: Perform topological sorting
partial_ordering = list(nx.topological_sort(G_transitive))    	
    	

#print("the user with min interactions",min_u)  	
#print("the user with maximum interactions is :",max_u)
    	


#print(newdict)
"""
OLD IMPL

list_of_sets = list(newdict.values())
course_list_for_seq = [list(s) for s in list_of_sets] 
model = PrefixSpan(course_list_for_seq)

# Mine frequent sequential patterns
patterns = model.frequent(5)  # Adjust the support threshold as needed. Now it returns all patterns with support threshold >=3.


frequent_pattern_list=[]
# Print the discovered patterns
#print("PATTERNS")
#print(patterns)
for pattern in patterns:
    if len(pattern[1])>1:
    	frequent_pattern_list.append(pattern[1])
    
    
#print("frequent pattern list",frequent_pattern_list)    
   





unique_vertices = set(vertex for pattern in frequent_pattern_list for vertex in pattern)

# Step 2: Create a directed graph
G = nx.DiGraph()

# Step 3: Add edges based on sequential patterns
for pattern in frequent_pattern_list:
    G.add_edges_from(zip(pattern[:-1], pattern[1:]))





# Step 4: Apply transitive closure (optional)
G_transitive = nx.transitive_closure(G)
'''

largest_wcc = max(nx.weakly_connected_components(G), key=len)

# Create a subgraph containing only the largest weakly connected component
subgraph = G.subgraph(largest_wcc)

# Specify the layout for the subgraph (you can choose a different layout if needed)
pos = nx.spring_layout(subgraph)

# Draw the subgraph with the specified layout
nx.draw(subgraph, pos, with_labels=True, node_color='red', font_color='black')
plt.savefig("sequence_order_G.jpg")
plt.show()
'''



# Print the edges of the graph
#print("Edges of the graph:")
#for edge in G.edges:
    #print(edge)

# Print the edges of the transitive closure graph
#print("\nEdges of the transitive closure graph:")
#for edge in G_transitive.edges:
    #print(edge)
'''  
pos = nx.circular_layout(G_transitive)  # You can choose a different layout if needed
nx.draw(G_transitive,pos, with_labels=True, node_color='red', font_color='black')
plt.savefig("sequence_order_G_trans.jpg")
plt.show()
'''

    
    
partial_ordering=topological_sort(G_transitive)  
#print("PARTIAL ORDERING")
#print(partial_ordering)

#print("len(partial ordering)",   len(partial_ordering))
    
    
"""    
    
    
    
   
        
        
#dummydict={"S1":{"C2","C3","C4"},"S2":{"C4","C5","C6"},"S3":{"C2","C3","C6","C7"}}
#print("dummydict",dummydict)
#SSH = hnx.Hypergraph(dummydict, static=True)
#hnx.draw(SSH)
#plt.savefig("index.jpg")

r_min = graph_data['learner_rating'].min()
r_max = graph_data['learner_rating'].max()

# Compute m_ui and 1 - m_ui
graph_data['mem_degree'] = (graph_data['learner_rating'] - r_min) / (r_max - r_min)
graph_data['non_degree'] = 1 - graph_data['mem_degree']


graph_data['P(i|u)'] = graph_data.groupby('learner_id')['course_id'].transform('count') / \
                       graph_data.groupby('learner_id')['course_id'].transform('size')

# Step 2: Group by user_id and calculate entropy for each user
def calculate_entropy(group):
    probabilities = group['P(i|u)']
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

user_entropy = graph_data.groupby('learner_id').apply(calculate_entropy).reset_index(name='entropy')

# Step 3: Merge entropy back into the original graph_data
graph_data = pd.merge(graph_data, user_entropy, on='learner_id', how='left')

# The 'graph_data' DataFrame now has a new column 'entropy'
#print(graph_data.head())
#finding the count of most frequent course
max_course_val=graph_data['course_id'].value_counts().nlargest(1)
max_course_count=max_course_val.values
#print("max_course_count",max_course_count[0])


#calculation membership changing parameter
mem=1/(max_course_count+1)



#updating membership and non membership degrees
'''
#CONSIDER RATING TOO
for index,row in graph_data.iterrows():
	cid=row["course_id"]
	for index,row in graph_data.iterrows():
		if row["course_id"]==cid:
			graph_data.loc[index,"mem_degree"]=row["mem_degree"]+mem
			graph_data.loc[index,"non_degree"]=row["non_degree"]-mem
			
			
'''





for index,row in graph_data.iterrows():
	graph_data.loc[index,"weight"]=row["learner_rating"]
	
	
#print("GRAPH_DATA")
#print(graph_data)


#create adjacency matrix1 of graph_data-edge vs nodes

cols = itemnum+1
rows = usernum+1

adj_mat=np.zeros(shape=(rows,cols))

for index,row in graph_data.iterrows():
	j=row["course_id"]
	i=row["learner_id"]
	w=row["weight"]
	adj_mat[i][j]=w
	
				

#f = open('user_item_num.json')
#data = json.load(f)
#no_of_clust=data['no_of_clust']
no_of_clust=usernum-5





topn=25


#**********************************************************************************************************************************************************************************************
#TESTING
#TOP-N PREDICTIONS
#Top n predictions for the users in test data 
data_reg=pd.read_csv("data_fuzzy.csv")#created using the code fuzzy_dataset.py
#data_reg=pd.read_csv("behaviourdata_fuzzy.csv")#created using the code fuzzy_dataset.py
data_reg=data_reg[['learner_id','course_id','learner_rating','n_course_avg_rating','n_Counts']]
#data_reg=data_reg[['n_course_avg_rating','n_Counts','n_instructr_perf','learner_rating']]
data_reg=data_reg.dropna()
X, y = data_reg[['n_course_avg_rating','n_Counts']], data_reg["learner_rating"]
poly = PolynomialFeatures(degree=5, include_bias=False)
poly_features = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.05, random_state=42)
poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train, y_train)
#print("X",X_test)
poly_reg_y_predicted = poly_reg_model.predict(X_test)

test_list= list(user_test.keys())
train_list=list(user_train.keys())
'''
test_list=test_data.learner_id.unique().tolist()
train_list=test_data.learner_id.unique().tolist()
'''
tp=0
fp=0
total=0
ndcg_tot=0
NDCG=0
HR=0
HR_partial=0
rmseval=0
p=0
r=0
best_ndcg=0
NDCG_partial=0
agg_p=0
agg_p_partial=0
agg_r=0
agg_r_partial=0
p_partial=0
r_partial=0
best_ndcg_partial=0
best_p=0
best_r=0
best_p_partial=0
best_r_partial=0
#model=rel(usernum,itemnum,df_train,graph_data)
print("unique User number", usernum)
print("unique item number", itemnum)





#creating course list original for all users as a dictionary
course_list_original_train = df_train.groupby('learner_id')['course_id'].apply(list).to_dict()

course_list_original_test = df_test.groupby('learner_id')['course_id'].apply(list).to_dict()

    
gnn_rel_rec(adj_mat, course_list_original_train,partial_ordering)

# Print the indices of the top n items
#print("Indices of top", n, "items with highest Q-values:", top_n_indices)



#course_list_original=df_test.loc[df_test['learner_id'] == unum, 'course_id'].tolist()

		
#course_list_ultimate=graph_data.loc[graph_data['learner_id']==unum,'course_id'].tolist()
'''	
for cour in course_list_original:
	if cour in top_n_indices:
		rank=top_n_indices.index(cour)
		if rank<topn:
			a=np.asarray([course_list_original[0:topn]])
			b=np.asarray([top_n_indices[0:topn]])	
			a=a.argsort().argsort()
			b=b.argsort().argsort()
			HR += 1
			if len(a[0])==len(b[0]):
				ndcg_tot=ndcg_tot+1
				NDCG += ndcg_score(a,b)
			else:
				print("error in NDCG calc")
					
				
	
	#cheking if the courses recommended are actually done by the learner
	# Flatten top_n_indices if it's multidimensional
	top_n_indices_flat = top_n_indices.flatten()

	top_n_indices_tuple = tuple(top_n_indices_flat)
	# Create a set from the tuple of indices
	top_n_indices_set = set(top_n_indices_tuple)
	#print("set1",set1)
	set1=top_n_indices_set
	set2=set(course_list_original)
	#print("set2",set2)
	
	if len(set1 & set2)>0:
		HR=HR+1
	#if len(set_partial & set2)>0:
		#HR_partial=HR_partial+1
	
	try:
		p=len(set1 & set2)/len(set1)
		r=len(set1 & set2)/len(set2)
		#agg_p=agg_p+p
		#agg_r=agg_r+r
	except:
		#print("division by zero")
		pass
	fn=len(set2.difference(set1))
	if p>best_p:
		best_p=p
		best_r=r
	
	
	

print("true positives=",tp)
print("false positives=",fp)
print("total=",total)

print("Precision=",precision/total)
print("recall=",recall/total)
print("F1 score=",F1/total)
print("HR=",HR/total)


print("FOR PARTIAL ORDER NEUTRO")
      
p_partial=p_partial/total
r_partial=r_partial/total
NDCG_partial=NDCG_partial/total
print("precision=	",best_p_partial)
print("recall=	",best_r_partial)
print("F1=	", 2*(best_p_partial*best_r_partial+.00001/(best_p_partial+best_r_partial+.001)))
print("NDCG=	",NDCG_partial)
print("best NDCG=	",best_ndcg_partial)
print("HR=	",HR_partial/total)   
print("Average Precision=",agg_p_partial/total)
print("Average Recall=",agg_r_partial/total)
'''


