"""
collect.py
"""
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

consumer_key = 'VA00WIXDSMZMiDFWmr8aylbDr'
consumer_secret = 'gFckFgd3kxzK284j28DQvniUoLeO5u8hVZRD1OtFGAgErmiXFu'
access_token = '3738267792-FEw1oJ1dqQQJG74MWeRv2fPf7Z6czrikpvUV81H'
access_token_secret = '595OxWzmCClyKvxvZKOAgFGhwdn85IkKNBeQZSP0SCnxH'

def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def robust_request(twitter, resource, params, max_tries=5):
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def get_tweets(twitter):
  tweets = robust_request(twitter, 'search/tweets', {'count':1000,'q':'cryptocurrency -filter:retweets'}, max_tries=5)
  
  f = open('tweets_demo.txt', 'w+', encoding='utf-8')
  for tweet in tweets:
    f.write(tweet['text']+"\n")
  f.close()

  return tweets

def get_users_and_friends(twitter, tweets):
  f = open('users_friends_demo.txt', 'w+')

  users_list = []
  for tweet in tweets:
    users_list.append(tweet['user']['screen_name'])
  
  user_friends_list = []
  for user in set(users_list):
    user_friends_req = robust_request(twitter, 'friends/ids', {'screen_name':user, 'count':2000}, max_tries=5)
    for friends in user_friends_req:
      user_friends_list.append(friends)
    f.write(user+":"+str(sorted(user_friends_list))+"\n")
    user_friends_list = []
  
  f.close()

def download_affin():
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')

    afinn = dict()

    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])
    return afinn
    pass

def separate_data(afinn):
   pos_words=set([key for key, value in afinn.items() if value>=0])
   f_pos = open('pos.txt', 'w+')
   for word in pos_words:
     f_pos.write(word+'\n')
   f_pos.close()

   neg_words = set([key for key, value in afinn.items() if value<0])
   f_neg = open('neg.txt', 'w+')
   for word in neg_words:
     f_neg.write(word+'\n')
   f_neg.close()
   pass

def main():
  afinn = download_affin()
  separate_data(afinn)
  twitter = get_twitter()
  tweets = get_tweets(twitter)
  get_users_and_friends(twitter, tweets)

if __name__ == '__main__':
    main()
