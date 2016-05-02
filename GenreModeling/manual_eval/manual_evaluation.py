import numpy as np
from scipy.spatial.distance import cosine
from urllib import unquote_plus
import time
import cPickle
import os
import sys
import textwrap

min_known_artists = 200


# little function to handle the annoying 80 char line wrapping on windows
def printer(string):
    out = textwrap.wrap(string)
    for line in out:
        print line

if __name__=='__main__':

    argv = sys.argv
    if len(argv)==2 and argv[1] == 'debug':
        debug = True
    else:
        debug = False


    #artist_features = np.load('artist_features_100.npy')

    # top_artist_bins = [1000,500,250,100,50,25]
    # last_bin = top_artist_bins[-1]
    # top_artist_bins = iter(top_artist_bins)
    # top = top_artist_bins.next()


   # def calc_dist(artist_a,artist_b):
   #     return cosine(artist_features[artist_dict[artist_a]],artist_features[artist_dict[artist_b]])

    artist_dict = {}
    with open('vocab_idx') as fin:
        for line in fin:
            line = line.strip().split()
            artist_dict[line[0]] = int(line[1])

    print '\n'*3

    username = ''
    while not username:
        printer("Please enter your name (or some unique identifier, *not* case sensitive, no spaces). If you're coming back to this after starting earlier, be sure to enter it exactly as you did before")
        username = raw_input("Response: ").lower()

    if os.path.exists(username+'_known_artists.pkl'):
        known_artists = cPickle.load(open(username+'_known_artists.pkl'))
    else:
        if os.path.exists(username+'_known_artists_INCOMPLETE.pkl'):
            idx,known_artists = cPickle.load(open(username+'_known_artists_INCOMPLETE.pkl'))
            print '\n'
            printer("Looks like you're partway through the recognition test. If this seems wrong, type STOP now to stop the program, and contact Jared. Otherwise just hit RETURN")
            result = raw_input("Response: ")
            if result.lower() == 'stop':
                sys.exit()
            else:
                print '\n'
                printer('Let\'s continue the recognition test. Remember, you\'ll see a bunch of musical artists, and for each you need to say if you are familiar with them or not. Enter "Z" for artists you ARE NOT familiar with and "/" for artists you ARE familiar with, follwed by RETURN.')
                raw_input('\nPress RETURN when you\'re ready to start')
        else:
            print '\n'
            printer("Ok, I don't see any history for you. If this seems wrong, type STOP now to stop the program, and contact Jared/Kyle. Otherwise just hit RETURN.")
            result = raw_input("Response: ")
            if result.lower() == 'stop':
                sys.exit()
            else:
                print '\n'
                printer('First things first: We\'re going to do a recognition test. You\'ll see a bunch of musical artists, and for each you need to say if you are familiar with them or not. Enter "Z" for artists you ARE NOT familiar with and "/" for artists you ARE familiar with, follwed by RETURN. At any time, type "EXIT" to quit and continue later.')
                raw_input('\nPress RETURN when you\'re ready to start.\n')
            known_artists = set()
            idx = 0

        last50 = []
        quit = False
        for i,artist in enumerate(sorted(artist_dict,key=artist_dict.get)):
            if quit:
                break
            if i<idx:
                continue
            accepted = False
            while not accepted:
                l = len(known_artists)
                if l>=min_known_artists:
                    query =  'Are you familiar with the musical artist "{}" ? (known artists: {})\nYou are now able to respond "skip" to skip to the next section, but please keep going if you can!)'.format(unquote_plus(artist),len(known_artists))
                else:
                    query =  'Are you familiar with the musical artist "{}" ? (known artists: {})'.format(unquote_plus(artist),len(known_artists))
                print query
                response = raw_input('Response ("Z" = NO, "/" = YES): ').lower()
                if response not in ('z','exit','/','quit','skip'):
                    print "Invalid response, please try again"
                    print '\n'
                elif response in ('exit','quit'):
                    cPickle.dump((i,known_artists),open(username+'_known_artists_INCOMPLETE.pkl','w'))
                    print '\n'
                    printer("Thanks! We've saved your progress and you can pick up where you left off by re-running the program.")
                    raw_input("Press RETURN to close the program.")
                    sys.exit()
                elif response == 'skip':
                    if len(known_artists)>=min_known_artists:
                        accepted = True
                        quit = True
                        printer("Alright - moving on similarity judgments.")
                        raw_input("Press RETURN to continue")
                    else:
                        printer("You need a set of at least {} known artists before moving on".format(min_known_artists))
                        print '\n'

                else:
                    accepted = True
                    if response == '/':
                        known_artists.add(artist)
                        last50.append(1)
                    else:
                        last50.append(0)
                    if len(last50)>50:
                        last50.pop(0)
                        prop_known = sum(last50)/50.
                        if prop_known < 0.5:
                            while response not in ('yes','no'):
                                printer("You knew less than half of the last 50 artists we showed you. You may now skip to the next part, but we would like you to keep going until you know at least {} artists. Would you like to skip now?".format(min_known_artists))
                                response = raw_input("Response (YES or NO): ").lower()
                            if response == 'yes':
                                quit = True
                            else:
                                last50 = []
            print '\n'

        cPickle.dump(known_artists,open(username+'_known_artists.pkl','w'))
        print 'Great! All done with the recognition test'


    sample_set = list(known_artists)
    #n = len(sample_set)
    #possible_combinations = factorial(n)/(factorial(3)*factorial(n-3))

    if os.path.exists(username+'_comps.pkl'):
        comps = cPickle.load(open(username+'_comps.pkl'))
    else:
        comps = set()

    print '\n'+'-'*50+'\n'
    print '\n'+'-'*50+'\n'
    done = 0
    done_total = len(comps)
    #unknown = 0.

    brk = False
    with open(username+'_log','a') as fout:
        while not brk:

            print 'Comparisons presented this session: {} (total: {})'.format(int(done),int(done_total))

            a,b,c=np.random.choice(sample_set,size=3,replace=False)
            good_comp_found = False
            while not good_comp_found:
                iden = tuple(sorted([artist_dict[artist] for artist in (a,b,c)]))
                if iden not in comps:
                    good_comp_found = True

            formatted_a = unquote_plus(a)
            formatted_b = unquote_plus(b)
            formatted_c = unquote_plus(c)
            #ab_dist = calc_dist(a,b)
            #bc_dist = calc_dist(b,c)
            accepted = False
            while not accepted:
                print '\n'
                printer('Remember, just hit RETURN if you really don\'t know or haven\'t heard of one of the artists (but if you\'re familiar with all of them, do go with your gut). Type "exit" to quit.')
                print '\n'
                #query =  'Is "{}" more like "{}" ("Z") or "{}" ("/")? Enter "=" for equally similar.\n'.format(formatted_b,formatted_a,formatted_c)
                query =  'Is "{}" more like "{}" ("Z") or "{}" ("/")? \n'.format(formatted_b,formatted_a,formatted_c)
                printer(query)
                result = raw_input("Response: ").lower()
                #if result not in ('z','/','','exit','=','quit'):
                if result not in ('z','/','','exit','quit'):
                    print "Invalid response, please try again"
                    printer(query)
                    result = raw_input("Response: ")
                elif result=='':
                    print "That's ok, moving on...."
                    fout.write('\t'.join(map(str,[time.strftime("%Y%m%d%H%M%S"),artist_dict[a],artist_dict[b],artist_dict[c],ab_dist,bc_dist,-1]))+'\n')
                    print '-'*50
                    accepted=True
                    comps.add(iden)
                    done+=1
                    done_total+=1
                    #unknown+=1
                elif result in ('exit','quit'):
                    brk=True
                    accepted = True
                    cPickle.dump(comps,open(username+'_comps.pkl','w'))
                    printer("Thanks! We've saved your progress and you can pick up where you left off by re-running the program. If you're all done, feel free to contact Jared and/or send him the file {}_log that's in the same directory as this program. You rock!!!".format(username))
                    raw_input("Press RETURN to close the program.")

                else:
                    parsed_response = {'z':a,'/':c,'=':'='}[result]
                    #response2 = raw_input('You said that {} is more like {} (and less like {}). Are you sure? (ENTER for yes, any other key to change response)'.format(b,parsed_response,other))
                    response2=None
                    if not response2:
                        accepted=True
                        comps.add(iden)
                        done+=1
                        done_total+=1

                        if result == 'z':
                            response = 'a'
                        elif result == '/':
                            response = 'c'

                        # if parsed_response == '=':
                        #     agree = 2
                        #     if debug:
                        #         print 'According to our model: {}<=>{} distance={:.2f},{}<=>{} distance={:.2f}'.format(
                        #             formatted_a,formatted_b,ab_dist,formatted_b,formatted_c,bc_dist)
                        # else:
                        #     if (ab_dist>bc_dist and parsed_response==c) or (ab_dist<bc_dist and parsed_response==a):
                        #         agree=1
                        #     else:
                        #         agree=0
                        #     if debug:
                        #         print '\nOur model {}! {}<=>{} distance={:.2f}, {}<=>{} distance={:.2f}'.format(
                        #             {1:'agrees',0:'disagrees'}[agree],formatted_a,formatted_b,ab_dist,formatted_b,formatted_c,bc_dist)

                        #fout.write('\t'.join(map(str,[time.strftime("%Y%m%d%H%M%S"),artist_dict[a],artist_dict[b],artist_dict[c],ab_dist,bc_dist,agree]))+'\n')
                        fout.write('\t'.join(map(str,[time.strftime("%Y%m%d%H%M%S"),artist_dict[a],artist_dict[b],artist_dict[c],response]))+'\n')

                        print '\n'+'-'*50+'\n'

                    else:
                        print "That's ok, do your thing"





