"""
sumarize.py
"""

def main():
    summary_f = open('summary.txt', 'w+')

    tweets_f = open('tweets.txt', 'r+', encoding='utf-8')
    summary_f.write('Number of messages collected: '+str(len(tweets_f.readlines()))+'\n\n')
    tweets_f.close()

    users_f = open('users_friends.txt', 'r+')
    summary_f.write('Number of users collected: '+str(len(users_f.readlines()))+'\n\n')
    users_f.close()

    cluster_f = open('clusters.txt', 'r+')
    for l in cluster_f.readlines():
        summary_f.write(l)
    cluster_f.close()

    classify_f = open('classifications.txt', 'r+')
    for l in classify_f.readlines():
        summary_f.write(l)
    classify_f.close()

    summary_f.close()

if __name__ == '__main__':
    main()