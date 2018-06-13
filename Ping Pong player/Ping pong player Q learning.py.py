
import numpy as np 
import time
from copy import deepcopy
import random
import math
import matplotlib.pyplot as plt

# -----------Parameters ------------------
C=50        #learning rate
gamma=0.7  #discount rate
epsilon =0.05
number_of_training_games=100000
number_of_test_games = 1000
# --------- Parameter's for exploration policy
N_thres=10
constant_reward =0.05

# --------- Variable initialization
Q = np.zeros( (12,12,2,3,12,3) )
N = np.zeros( (12,12,2,3,12,3) )
Policy = np.zeros( (12,12,2,3,12) )
time_begin=time.time()

paddlehit_vs_gamenum = np.zeros( number_of_training_games +1 )
gamenum = np.linspace(1, number_of_training_games+1, number_of_training_games +1 )


def ball_projection_to_paddle(cont_state_prime):
	proj=np.zeros(2)
	proj[0]= ( cont_state_prime[4] - cont_state_prime[1] ) * cont_state_prime[2]/ (1 -cont_state_prime[0]) + cont_state_prime[1] 
	proj[1]= ( cont_state_prime[4] +0.2 - cont_state_prime[1] ) * cont_state_prime[2]/ (1 -cont_state_prime[0]) + cont_state_prime[1]
	return proj


def apply_action(cont_state, action):
	# create variable for the updated state
	cont_state_prime = deepcopy( cont_state)

	# move the paddle
	cont_state_prime[4] += action
	# make sure paddle can't go pass boundary
	if( cont_state_prime[4] > 0.8):
		cont_state_prime[4]=0.8
	if( cont_state_prime[4] <0):
		cont_state_prime[4]=0
	down_bound, top_bound = tuple( ball_projection_to_paddle(cont_state_prime) )

	# move the ball
	cont_state_prime[0] += cont_state[2]
	cont_state_prime[1] += cont_state[3]

	# if the ball hits the left wall
	if(cont_state_prime[0] <= 0 ):
		cont_state_prime[0] = -1* cont_state_prime[0]
		cont_state_prime[2] = -1* cont_state_prime[2] 
	
	# if the ball the hits paddle at x=1
	if( cont_state_prime[0] >= 1 and cont_state_prime[1]>= down_bound and cont_state_prime[1]<=top_bound ):
		cont_state_prime[0] = 2 - cont_state_prime[0]
		cont_state_prime[2] = -1* cont_state_prime[2] + np.random.uniform(low= -0.015, high= 0.015)
		cont_state_prime[3] += np.random.uniform(low=-0.03, high=0.03)

	# if the ball hits the up wall
	if( cont_state_prime[1] <=0):
		cont_state_prime[1] = -1*cont_state_prime[1]
		cont_state_prime[3] = -1*cont_state_prime[3] 

	# if the ball hits the down wall
	if( cont_state_prime[1] >= 1):
		cont_state_prime[1] = 2 - cont_state_prime[1]
		cont_state_prime[3] = -1*cont_state_prime[3] 

	# make sure |v_x| >0.03
	if( np.absolute( cont_state_prime[2] ) < 0.03): 
		cont_state_prime[2] =0.03* np.sign( cont_state_prime[2] )

	return cont_state_prime


def cont_to_disc( cont_state):
	disc_state = np.zeros( 5, dtype=int)
	vy_zero_threshold = 0.2

	disc_state[0]= min( int( 12* cont_state[0] -0.0001) ,11)
	disc_state[1]= min( int( 12* cont_state[1] -0.0001) ,11)

	if( cont_state[2] <0):
		disc_state[2]=0
	else:
		disc_state[2]=1
	    
	if( np.absolute( cont_state[3] ) < vy_zero_threshold ):
		disc_state[3]= 1
	else:
		disc_state[3]= np.sign( cont_state[3] )  +1
	    
	disc_state[4]= int( 12* cont_state[4]/0.8 -0.0001)

	return disc_state

def get_Q_value(d_state, action):
	action = int( action/0.04 + 1 )
	return Q [tuple(d_state)] [action]

def get_N_value(d_state, action):
	action = int( action/0.04 + 1 )
	return N [tuple(d_state)] [action]

def get_Policy_value(d_state):
	return Policy [tuple(d_state)]

def exploration_policy2(d_state):
	all_action = [-0.04, 0, 0.04]
	action = random.choice( all_action  )

	random_num = np.random.uniform(low=0, high=1)

	# Take greedy action with epsilon probability
	if( random_num > epsilon):
		Max= -math.inf
		for i in all_action:
			Qval= get_Q_value(d_state, i)
			if( Qval > Max):
				Max= Qval
				action=i
	return action

def exploration_policy(d_state):
	all_action = [-0.04, 0, 0.04]
	Max= -math.inf
	argmax = -math.inf

	for action in all_action:
		N_sa = get_N_value(d_state, action)
		if( N_sa < N_thres ): 
			reward = constant_reward
			if( reward> Max ):
				Max = reward
				argmax=action
		else:
			reward = get_Q_value(d_state, action)
			if( reward> Max ):
				Max = reward
				argmax=action
	return argmax


def game_check(cont_state):
	game_ended=False
	if ( cont_state[0] > 1 ):
		game_ended= True
	return game_ended

def Q_update(disc_state, disc_state_prime, cont_state, cont_state_prime, a):
	Q_sa_old = get_Q_value(disc_state,a)
	N_sa = get_N_value(disc_state,a)

	# Calculate reward(disc_state)
	Game_ended = game_check(cont_state_prime)
	Paddle_hit = paddle_hit(cont_state,a)
	reward=0
	if( Game_ended== True):
		reward=-1
	if( Paddle_hit== True ):
		reward=1

	Q_s_prime_a_max = max( get_Q_value(disc_state_prime,-0.04), get_Q_value(disc_state_prime, 0.04), get_Q_value(disc_state_prime,0) )

	Q [tuple(disc_state)] [ int( a/0.04 +1) ] = Q_sa_old + C/(C + N_sa)* (reward + gamma*Q_s_prime_a_max - Q_sa_old)
	return 0

def paddle_hit(disc_state, action):
	paddle_hit=False

	# move the paddle
	p_y = disc_state[4] + action

	# find ball projection to the paddle
	down_bound, top_bound = ball_projection_to_paddle(disc_state)

	# move the ball
	b_x = disc_state[0]+disc_state[2]
	b_y = disc_state[1]+disc_state[3]

	#check if ball hit the paddle
	if( b_x>=1 and b_y>=down_bound and b_y<= top_bound  ):
		paddle_hit=True
	return paddle_hit

# --------------------   Learning
# for i in range( number_of_training_games):
for i in range( number_of_training_games ):
	cont_state=np.array( [ 0.5,0.5, 0.03, 0.01, 0.4 ] )
	game_ended = False

	# oldpadel=0
	game_bounce_number=0
	while( not game_ended):

		# choose action based on disc_state and the policy
		disc_state = cont_to_disc(cont_state)
		a = exploration_policy2(disc_state)

		# update the policy
		Policy[ tuple(disc_state) ] = a

		# update frequency of (disc_state, action) pair
		N [tuple(disc_state)] [ int( a /0.04 +1) ] +=1

		# increment total_padel_hit, if the ball hit the paddle
		if( paddle_hit(cont_state,a) ):
			# total_padel_hit+=1
			game_bounce_number+=1

		# update cont_state based on action
		cont_state_prime = apply_action(cont_state, a)
		disc_state_prime = cont_to_disc(cont_state_prime)
		

		# update the Q-val of current state
		Q_update(disc_state, disc_state_prime, cont_state, cont_state_prime, a )

		# has the game ended?
		game_ended = game_check(cont_state_prime)

		#update the state
		cont_state= deepcopy( cont_state_prime )

	paddlehit_vs_gamenum[i+1] = paddlehit_vs_gamenum[i] + 1/(i+1)* ( game_bounce_number - paddlehit_vs_gamenum[i] )
	print( " game: ", i, "bounces: ", game_bounce_number)


##------------------  Testing  -------------------------
total_padel_hit=0
for i in range( number_of_test_games):
	cont_state= np.array( [ 0.5,0.5, 0.03, 0.01, 0.4 ] )

	game_ended = False
	game_bounce_number=0
	while(not game_ended ):
		# choose action based on disc_state and the policy
		disc_state = cont_to_disc(cont_state)
		a = get_Policy_value(disc_state)

		# increment total_padel_hit, if the ball hit the paddle
		if( paddle_hit(cont_state,a) ):
			total_padel_hit+=1
			game_bounce_number+=1
		# update cont_state based on action
		cont_state_prime = apply_action(cont_state, a)
		disc_state_prime = cont_to_disc(cont_state_prime)

		# has the game ended?
		game_ended = game_check(cont_state_prime)

		# update the state
		cont_state = deepcopy( cont_state_prime)


time_end = time.time()
time_elapsed= time_end - time_begin

print( "------------------- Game Status -----------------------")
print(" Average padel hit per game (in " + str( number_of_test_games) + "test games): ", total_padel_hit/number_of_test_games)
print( "Time elapsed: ", time_elapsed)

#-------------------------------------------------------------------------------------------


##--------------- Plotting-------------------
f,ax = plt.subplots()
ax.plot(gamenum, paddlehit_vs_gamenum, 'b', label='Average Bounce')
plt.ylabel('average number of rebounce')
plt.xlabel('the number of games')
plt.title('Alpha = ' + str( C) + '/(' + str( C ) + '+N(s,a)), Gamma = ' + str(gamma) + ', Epsilon = ' + str(epsilon))
plt.legend(loc='best')
plt.ylim(0,14)
plt.grid(True)
plt.show()







# ---------------- Archive
def print_state( disc_state):
	a = [ " "] *12
	board = np.array( [a]*12 )

	board[ disc_state[4]] [11] = "|"
	board[ disc_state[4]+1][11]= "|"

	board[ disc_state[1] ] [ disc_state[0] ] = "b"
	print("----------------------------------------------")
	print(board)
	return 0
# for i in range(2):
# 	cont_state= np.array( [ 0.5,0.5, 0.03, 0.01, 0.4 ] )
# 	game_ended = False
# 	while( not game_ended):
# 		# choose action based on disc_state and the policy
# 		disc_state = cont_to_disc(cont_state)
# 		a = get_Policy_value(disc_state)

# 		# increment total_padel_hit, if the ball hit the paddle
# 		if( paddle_hit(cont_state,a) ):
# 			total_padel_hit+=1

# 		# update cont_state based on action
# 		cont_state_prime = apply_action(cont_state, a)
# 		disc_state_prime = cont_to_disc(cont_state_prime)


# 		# has the game ended?
# 		game_ended = game_check(cont_state_prime)

# 		# update the state
# 		cont_state = deepcopy( cont_state_prime )
# 		print_state(disc_state_prime)
# 		time.sleep(0.3)