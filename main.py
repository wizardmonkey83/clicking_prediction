import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
from collections import defaultdict

## loads the .csv file into a pandas dataframe
df = pd.read_csv('/path/to/your/clicks_train.csv')

## 'df.groupby('display_id')['ad_id'].apply(list)' gathers all of the 'ad_id' values in a 'display_id' session and puts them into a list.
sessions = df.groupby('display_id')['ad_id'].apply(list).reset_index()

## gathers all of the 'clicked' values in a 'display_id' session into a list
sessions_clicked = df.groupby('display_id')['clicked'].apply(list).reset_index()

## two lists for each session are combines
sessions['clicked'] = sessions_clicked['clicked']

## initializes a nested dictionary where each key points to another dictionary of counts. this stores the counts of transitions between each 'ad_id'
transition_counts = defaultdict(lambda: defaultdict(int))


## loops through each row in 'sessions'. '_' ignores the row's index, 'row' represents the information stored in the row. 
for _, row in sessions.iterrows():
    ## extracts the sequence of 'ad_id' in the current session.
    ad_sequence = row['ad_id']
    ## 'range(len(ad_sequence) - 1)' gets the total number of ads in the sequence. 
    for i in range(len(ad_sequence) - 1):
        ## represents the current ad being analyzed
        current_ad = ad_sequence[i]
        ## represents the ad that comes after the current ad
        next_ad = ad_sequence[i + 1]
        ## transitions between 'ad_id' as it adds a 1 to the numerical value representing the ads index
        transition_counts[current_ad][next_ad] += 1
## that loop was meant to create a dictionary of transitions between ads in sessions for the entire dataset


## contains all of the unique ads from 'transition_counts'
ads = list(transition_counts.keys())
## finds the number of unique ads, which will define the dimensions of the transition matrix
num_ads = len(ads)
## initializes an empty transition matrix with dimensions --> num_ads x num_ads
## each row represents a starting ad, and each column represents a potential next ad
transition_matrix = np.zeros((num_ads, num_ads))
## fills in the transition matrix by calculating probabilities for each ad transition
for i, ad in enumerate(ads):
    ## retrieves the dictionary of counts for transitions from the current ad to other ads
    next_ads = transition_counts[ad]
    ## calculates the total number of transitions originating from a certain ad
    ## this value will be used to normalize each transition count to a probability
    total_transitions = sum(next_ads.values())
    ## iterates through each unique ad to populate the transition probabilities
    for j, next_ad in enumerate(ads):
        ## calculates the probability of transitioning from ad to next_ad by dividing 
        ## the count of transitions from ad to next_ad by the total_transitions
        ## if no transitions to next_ad exist, it defaults to 0
        transition_matrix[i, j] = next_ads.get(next_ad, 0) / total_transitions

## sets initial probabilities for the hidden states (ads) assuming a uniform distribution
## each ad has an equal probability of being the starting state
initial_distribution = tfp.distributions.Categorical(probs=[1/num_ads] * num_ads)

## defines the transition distribution using the transition matrix
## each row in the matrix represents the probability of transitioning from one ad to each other ad
transition_distribution = tfp.distributions.Categorical(probs=transition_matrix)

## defines the observation distribution, assuming direct observation of each ad
## uses an identity matrix so that each ad is "observed" directly with no noise
## each ad has a probability of 1 of observing itself, representing a straightforward mapping
observation_distribution = tfp.distributions.Categorical(probs=np.eye(num_ads))

## creates the hidden markov model using tensorflow probability with the specified distributions
## initial_distribution: defines starting probabilities for each ad
## transition_distribution: defines the probability of moving from one ad to another
## observation_distribution: defines the probability of observing each ad, assuming direct observation
## num_steps: specifies the number of time steps for which we want to observe the sequence
num_steps = 10  ## defines the number of time steps to observe
model = tfp.distributions.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=num_steps
)

## calculates the expected sequence of states over num_steps time steps
## this provides the most likely sequence of transitions based on the model
mean = model.mean()

## opens a tensorflow session to evaluate the mean, necessary for tensorflow probability calculations
with tf.compat.v1.Session() as sess:
    ## prints a description of the output
    print("Expected sequence of ad transitions:")
    
    ## runs the tensorflow session to calculate the mean and prints the expected sequence of transitions
    print(sess.run(mean))
