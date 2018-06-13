import numpy as np
import time
import matplotlib.pyplot as plt

class State(object):
    value=0
    White_Positions=[]
    Black_Positions=[]
    Extreme_child=0
    subTree_size=0
    
    def __init__(self, White, Black):
        self.White_Positions=White
        self.Black_Positions=Black
        return



#################################################################################
## ------------- Heuristic functions ---------------------------
#################################################################################

def DH1(Node):

    number_of_own_pieces= len( Node.White_Positions)
    solution= 2*number_of_own_pieces +np.random.rand()
    return solution

def OH1(Node):

    number_of_enemy_pieces= len( Node.Black_Positions)
    solution= 2*(30-number_of_enemy_pieces) +np.random.rand()    
    return solution

def DH2(Node):

    number_of_own_pieces= len( Node.White_Positions)

    #calculating enemy_position_value 
    enemy_position_value =0
    Black_weights = [1,1,1,1,1,1,5,10]
    for blackx, blacky in Node.Black_Positions:
        enemy_position_value += Black_weights[blacky-1]

    #calculating enemy_position_value 
    own_position_value =0
    White_weights = [1,1,1,1,1,1,5,10]
    for whitex,whitey in Node.White_Positions:
        own_position_value += White_weights[whitey-1]

    # solution = 1000 + number_of_own_pieces - enemy_position_value + np.random.rand()
    solution = 10000 - enemy_position_value + own_position_value +np.random.rand()
    return solution


def OH2(Node):

    number_of_enemy_pieces= len( Node.Black_Positions)

    #calculating enemy_position_value 
    own_position_value =0
    White_weights = [1,1,1,1,1,3,5,10]
    for whitex,whitey in Node.White_Positions:
        own_position_value += White_weights[whitey-1]

    solution = 10000 -number_of_enemy_pieces + own_position_value +np.random.rand()
    return solution

#################################################################################
## ------------- Minimax search ---------------------------
#################################################################################

# Returns max value of CurNode. 
# CurNode.Extreme_child and CurNode.Child_States are added in process.
def Max_value(CurNode, heuristic, level, level_threshold):

    maxval=-9999999
    CurNode.subTree_size=1

    # For each White peice, consider all 3 moves
    for x,y in CurNode.White_Positions: 
        # Left-Up diagonal move
        x_new = x-1
        y_new = y-1
        if( Is_white_legal(x,y,x_new,y_new, CurNode) ):
            ChildA= Add_ChildState_Max(x,y, x_new,y_new, CurNode, heuristic, "minimax", level, level_threshold,0,0)  
            CurNode.subTree_size +=ChildA.subTree_size

            if( ChildA.value > maxval):
                maxval= ChildA.value
                CurNode.Extreme_child= ChildA

        # Right-up diagonal move
        x_new= x+1
        y_new= y-1
        if( Is_white_legal(x,y,x_new,y_new, CurNode) ):
            ChildB= Add_ChildState_Max(x,y, x_new,y_new, CurNode, heuristic, "minimax", level, level_threshold,0,0)
            CurNode.subTree_size += ChildB.subTree_size

            if( ChildB.value > maxval):
                maxval= ChildB.value
                CurNode.Extreme_child= ChildB

        #Up move
        x_new= x
        y_new= y-1
        if( Is_white_legal(x,y,x_new,y_new, CurNode) ):
            ChildC= Add_ChildState_Max(x,y, x_new,y_new, CurNode, heuristic, "minimax", level, level_threshold,0,0)
            CurNode.subTree_size += ChildC.subTree_size

            if( ChildC.value > maxval):
                maxval= ChildC.value
                CurNode.Extreme_child= ChildC

    return maxval

# White player uses Max
# If move is legal, returns newChild() 
# If move is not legal, return CurNode.
def Add_ChildState_Max(x_old,y_old,x_new,y_new, CurNode, heuristic, searchtype, level, level_threshold, alpha, beta):

    # Create new_White_Positions
    new_White_Positions = CurNode.White_Positions[:]

    # Find and update whitepiece position
    iteration=0
    for white_x,white_y in CurNode.White_Positions:
        if( x_old==white_x and y_old==white_y):
            new_White_Positions[iteration]= (x_new,y_new)
        iteration+=1


    # Create new_Black_Positions, with captured peice removed
    new_Black_Positions=CurNode.Black_Positions[:]
    iteration=0
    for black_x,black_y in CurNode.Black_Positions:
        if( x_new==black_x and y_new==black_y):
            del new_Black_Positions[iteration] 
        iteration+=1

    # Create newChild 
    newChild = State(new_White_Positions, new_Black_Positions)
    
    #Find newChild.value
    if( level < level_threshold):
        if( searchtype=="alphabeta"):
            newChild.value= AlphaBeta_Min_value(newChild, heuristic, level+1, level_threshold, alpha, beta)
        else: #minimax search
            newChild.value= Min_value(newChild, heuristic, level+1, level_threshold)

    else:
        if( heuristic=="OH1"):
            newChild.value= OH1(newChild)
        elif( heuristic== "OH2"):
            newChild.value= OH2(newChild)
        elif( heuristic=="DH1"):
            newChild.value= DH1(newChild)
        else:
            newChild.value= DH2(newChild)

    return newChild


# Returns min value of CurNode. 
# CurNode.Extreme_child and CurNode.Child_States are added in process.
def Min_value(CurNode, heuristic, level, level_threshold):
    minval = 9999999
    CurNode.subTree_size = 1
    # For each Black peice, consider all 3 moves
    for x,y in CurNode.Black_Positions: 
        # Left-Down diagonal move
        x_new = x-1
        y_new = y+1
        if( Is_black_legal(x,y,x_new,y_new, CurNode) ):
            ChildA = Add_ChildState_Min(x,y, x_new,y_new, CurNode, heuristic, "minimax", level, level_threshold,0,0)
            CurNode.subTree_size +=ChildA.subTree_size

            if( ChildA.value < minval):
                minval= ChildA.value
                CurNode.Extreme_child= ChildA   

        # Right-up diagonal move
        x_new= x+1
        y_new= y+1
        if( Is_black_legal(x,y,x_new,y_new, CurNode) ):
            ChildB = Add_ChildState_Min(x,y, x_new,y_new, CurNode, heuristic, "minimax", level, level_threshold,0,0)
            CurNode.subTree_size += ChildB.subTree_size

            if( ChildB.value < minval):
                minval= ChildB.value
                CurNode.Extreme_child= ChildB   

        #Down move
        x_new= x
        y_new= y+1
        if( Is_black_legal(x,y,x_new,y_new, CurNode) ):
            ChildC = Add_ChildState_Min(x,y, x_new,y_new, CurNode, heuristic, "minimax", level, level_threshold,0,0)
            CurNode.subTree_size += ChildC.subTree_size

            if( ChildC.value < minval):
                minval= ChildC.value
                CurNode.Extreme_child= ChildC

    return minval




# Black player uses Minv_value()
# If move is legal, returns newChild() 
# If move is not legal, return CurNode.
def Add_ChildState_Min(x_old,y_old,x_new,y_new, CurNode2, heuristic, searchtype, level, level_threshold, alpha, beta):

    # Create new_Black_Positions
    new_Black_Positions = CurNode2.Black_Positions[:]

    # Find and update blackpiece position
    iteration=0 
    for black_x, black_y in CurNode2.Black_Positions:
        if( x_old==black_x and y_old==black_y):
            new_Black_Positions[iteration]= (x_new,y_new)
        iteration+=1

    # Create new_White_Positions, remove captured white peice 
    new_White_Positions=CurNode2.White_Positions[:]
    iteration=0
    for whitex, whitey in CurNode2.White_Positions:
        if( x_new==whitex and y_new==whitey):
            del new_White_Positions[iteration] 
        iteration+=1

    # Create newChild 
    newChild = State(new_White_Positions, new_Black_Positions)

    #Find newChild.value
    if( level < level_threshold):
        if(searchtype=="alphabeta"):
            newChild.value= AlphaBeta_Max_value(newChild, heuristic, level+1, level_threshold, alpha, beta)
        else:
            newChild.value= Max_value(newChild, heuristic, level+1, level_threshold)
        
    else:
        if( heuristic=="OH1"):
            newChild.value= OH1(newChild)
        elif( heuristic== "OH2"):
            newChild.value= OH2(newChild)
        elif( heuristic=="DH1"):
            newChild.value= DH1(newChild)
        else:
            newChild.value= DH2(newChild)    

    return newChild

#################################################################################
## ------------- Alpha Beta Search ---------------------------
#################################################################################

def AlphaBeta_Min_value(CurNode, heuristic, level, level_threshold, alpha, beta):

    minval=9999999
    CurNode.subTree_size=1
    # For each Black peice, consider all 3 moves
    for x,y in CurNode.Black_Positions: 
        # Left-Down diagonal move
        x_new = x-1
        y_new = y+1
        if( Is_black_legal(x,y,x_new,y_new, CurNode) ):
            ChildA = Add_ChildState_Min(x,y, x_new,y_new, CurNode, heuristic, "alphabeta", level, level_threshold,alpha, beta)
            CurNode.subTree_size += ChildA.subTree_size

            if( ChildA.value < minval):
                minval= ChildA.value
                CurNode.Extreme_child= ChildA   

            if( minval <=alpha ):
                return minval
            beta = min(beta, minval)


        # Right-up diagonal move
        x_new= x+1
        y_new= y+1
        if( Is_black_legal(x,y,x_new,y_new, CurNode) ):
            ChildB = Add_ChildState_Min(x,y, x_new,y_new, CurNode, heuristic, "alphabeta", level, level_threshold, alpha, beta)
            CurNode.subTree_size += ChildB.subTree_size

            if( ChildB.value < minval):
                minval= ChildB.value
                CurNode.Extreme_child= ChildB   

            if( minval <=alpha):
                return minval
            beta = min(beta, minval)

        #Down move
        x_new= x
        y_new= y+1
        if( Is_black_legal(x,y,x_new,y_new, CurNode) ):
            ChildC = Add_ChildState_Min(x,y, x_new,y_new, CurNode, heuristic, "alphabeta", level, level_threshold, alpha, beta)
            CurNode.subTree_size +=ChildC.subTree_size

            if( ChildC.value < minval):
                minval= ChildC.value
                CurNode.Extreme_child= ChildC
            
            if( minval <=alpha):
                return minval
            beta = min(beta, minval)

    return minval



def AlphaBeta_Max_value(CurNode, heuristic, level, level_threshold, alpha, beta):

    maxval=-9999999
    CurNode.subTree_size=1
    # For each White peice, consider all 3 moves
    for x,y in CurNode.White_Positions: 
        # Left-Up diagonal move
        x_new = x-1
        y_new = y-1
        if( Is_white_legal(x,y,x_new,y_new, CurNode) ):
            ChildA= Add_ChildState_Max(x,y, x_new,y_new, CurNode, heuristic, "alphabeta", level, level_threshold, alpha, beta)
            CurNode.subTree_size +=ChildA.subTree_size

            if( ChildA.value > maxval):
                maxval= ChildA.value
                CurNode.Extreme_child= ChildA

            if( maxval >=beta):
                return maxval #end the forloop. 
            alpha= max( alpha, maxval)

        # Right-up diagonal move
        x_new= x+1
        y_new= y-1
        if( Is_white_legal(x,y,x_new,y_new, CurNode) ):
            ChildB= Add_ChildState_Max(x,y, x_new,y_new, CurNode, heuristic,"alphabeta",level, level_threshold, alpha, beta)
            CurNode.subTree_size +=ChildB.subTree_size
            if( ChildB.value > maxval):
                maxval= ChildB.value
                CurNode.Extreme_child= ChildB

            if( maxval >=beta):
                return maxval #end the forloop. 
            alpha= max( alpha, maxval)

        #Up move
        x_new= x
        y_new= y-1
        if( Is_white_legal(x,y,x_new,y_new, CurNode) ):
            ChildC= Add_ChildState_Max(x,y, x_new,y_new, CurNode, heuristic,"alphabeta", level, level_threshold, alpha, beta)
            CurNode.subTree_size+=ChildC.subTree_size
            if( ChildC.value > maxval):
                maxval= ChildC.value
                CurNode.Extreme_child= ChildC

            if( maxval >=beta):
                return maxval #end the forloop. 
            alpha= max( alpha, maxval)

    return maxval


#################################################################################
## ------------- Helper functions ---------------------------
#################################################################################
def Is_black_legal(oldx,oldy,newx,newy, Node):
    solution=True

    # Within bound?
    if(newx<1 or newx>8 or newy<1 or newy>8 ):
        solution=False
    # Check for illegal overlap with black
    for blackx, blacky in Node.Black_Positions:
        if( newx==blackx and newy==blacky):
            solution=False
    #Does black move down and illegaly capture white
    if( oldx==newx and oldy+1==newy):
        for whitex, whitey in Node.White_Positions:
            if(newx==whitex and newy==whitey):
                solution=False

    return solution


def Is_white_legal(oldx, oldy, newx, newy, Node):
    solution=True

    # Within bound?
    if( newx<1 or newx>8 or newy<1 or newy>8):
        solution=False
    # Check illegal overlap with white
    for white_x, white_y in Node.White_Positions:
        if( newx==white_x and newy==white_y):
            solution=False
    # Does white move up and overlap with black
    if(oldx==newx and oldy -1==newy):
        for black_x,black_y in Node.Black_Positions:
            if(newx==black_x and newy==black_y):
                solution = False

    return solution

def Initial_Node():
    White=[]
    Black=[]
    for x in range(1,9):
        for y in range(1,3):
            Black.append( (x,y) )

    for x in range(1,9):
        for y in range(7,9):
            White.append( (x,y) )

    StartNode=State(White, Black)
    return StartNode

def game_ended(Node):
    solution=False
    if( len(Node.White_Positions)==0 or len(Node.Black_Positions)==0 ):
        solution=True
        return solution

    for whitex, whitey in Node.White_Positions:
        if( whitey==1):
            solution=True
            return solution
    for blackx, blacky in Node.Black_Positions:
        if( blacky==8):
            solution=True
            return solution

    return solution

#Print state of board as string
def text_print_state(Node):
    A = np.array( ["."]*8 )
    B=np.array( [ A for i in range(8)] )

    if(len(Node.White_Positions) !=0 ):
        for wx,wy in Node.White_Positions:
            B[wy-1][wx-1]= "W"
    if( len(Node.Black_Positions) !=0):
        for bx,by in Node.Black_Positions:
            B[by-1][bx-1]= "B"

    print( "------------------------------")
    for i in range(8):
        newline= " ".join( B[i] )
        newline +="\n"
        print(newline)
    return

# Prints name of winner         
def who_won( Node):
    if( game_ended(Node) == True):
        #Has a white piece reached y=1
        White_reached=False
        for whitex,whitey in Node.White_Positions:
            if(whitey == 1):
                White_reached=True
        # All blacks captured or a White piece reached y=1
        if( len(Node.Black_Positions)==0 or White_reached):
            print( "White Player wins!")

        # Has a black piece reached y=8
        Black_reached=False
        for blackx, blacky in Node.Black_Positions:
            if(blacky==8):
                Black_reached=True
        # All White captured or a black piece reached y=8
        if( len(Node.White_Positions)==0 or Black_reached==True):
            print( "Black Player wins!")
    else:
        print("Game has not ended")
    return

def plot_state(Node):
    if( len(Node.White_Positions ) != 0 ):
        plt.plot( *zip(*Node.White_Positions), "ro")
    if( len(Node.Black_Positions) !=0 ):
        plt.plot( *zip(*Node.Black_Positions), "bs")
    print( "White is red. Black is Blue")

    return

################################################################################
# ------------- Games function ---------------------------
################################################################################

def Game(searchtype_white, heuristic_white, searchtype_black, heuristic_black):
    CurNode = Initial_Node()
    player= "white"
    Total_node_expanded_white=0
    Total_node_expanded_black=0
    Total_Time_white=0
    Total_Time_black=0
    Total_number_of_moves_white=0
    Total_number_of_moves_black=0
    
    while( game_ended(CurNode) ==False ):
        if( player=="white" ):
            time_begin= time.time()
            if( searchtype_white=="alphabeta"):
                CurNode.value = AlphaBeta_Max_value(CurNode, heuristic_white, 0,3, -999999, 99999) #setting level=0, level_threshold=5
            else:
                CurNode.value = Max_value(CurNode, heuristic_white,0,2)
            time_end= time.time()

            Total_Time_white += time_end - time_begin
            Total_node_expanded_white += CurNode.subTree_size
            Total_number_of_moves_white+=1


            #go to next state
            CurNode = CurNode.Extreme_child
            player = "black"
            text_print_state(CurNode)
        else:
            
            time_begin= time.time()
            if(searchtype_black=="alphabeta"):
                CurNode.value = AlphaBeta_Min_value( CurNode, heuristic_black, 0, 3, -999999, 99999)
            else:
                CurNode.value = Min_value(CurNode, heuristic_white, 0,2)
            time_end = time.time()

            Total_Time_black += time_end - time_begin
            Total_node_expanded_black += CurNode.subTree_size
            Total_number_of_moves_black+=1

            #go to next state
            CurNode = CurNode.Extreme_child
            player = "white"
            text_print_state(CurNode)
    #end of while loop

    White_Captured = 16 - len( CurNode.White_Positions )
    Black_Captured = 16 - len( CurNode.Black_Positions )

    White_Avg_Nodes_Expanded_permove = Total_node_expanded_white/Total_number_of_moves_white
    Black_Avg_Nodes_Expanded_permove = Total_node_expanded_black/Total_number_of_moves_black

    White_avg_time_permove = Total_Time_white/Total_number_of_moves_white
    Black_avg_time_permove = Total_Time_black/Total_number_of_moves_black

    print( "Number of White moves: ", Total_number_of_moves_white)
    print( "Number of Black moves: ", Total_number_of_moves_black)
    print( "White: Avg nodes expanded per move: ", White_Avg_Nodes_Expanded_permove)
    print( "Black: Avg nodes expanded per move: ", Black_Avg_Nodes_Expanded_permove)
    print( "White: Avg time per move: ", White_avg_time_permove)
    print( "Black: Avg time per move: ", Black_avg_time_permove)
    print( "Number of White captured: ", White_Captured)
    print( "Number of Black captured: ", Black_Captured)
    print( "Total time of Game: ", Total_Time_black+Total_Time_white)
    # plot_state(CurNode)
    # text_print_state(CurNode)
    who_won(CurNode)


    return

# #################################################################################
# ## ------------- Game Simulations ---------------------------
# #################################################################################

##-------- Required game (uncomment desired games)
#Game("minimax", "OH1", "alphabeta", "OH1")
#Game("alphabeta", "OH2", "alphabeta", "DH1")
#Game("alphabeta", "DH2", "alphabeta", "OH1")
Game( "alphabeta", "OH2","alphabeta", "OH1")
#Game( "alphabeta", "DH2","alphabeta", "DH1")
#Game( "alphabeta", "OH2","alphabeta","DH2")



#Extra games
#Game( "alphabeta", "DH1","alphabeta", "DH2")
#Game( "alphabeta", "DH1","alphabeta", "DH2")
# Game("alphabeta", "OH1", "minimax", "OH1")
# Game( "alphabeta", "OH2","minimax", "OH1")
# Game( "alphabeta", "DH2","minimax", "DH1")
# Game("minimax", "OH2", "minimax", "OH1")
#Game("minimax", "DH2", "minimax", "DH1")
#Game("minimax", "OH1", "minimax", "DH1")
# Game("minimax", "OH1", "minimax", "DH1")

# Test 1 move
# Start = Initial_Node()
# time_begin=time.time()
# Start.value = AlphaBeta_Max_value(Start, "alphabeta", 0, 4, -99999, 99999)
# time_end = time.time()
# time_took= time_end - time_begin
# print("Time elapsed: " , time_took)
# text_print_state(Start.Extreme_child)



# plot_state(Start.Extreme_child.Extreme_child.Extreme_child)
# plot_state(Start.Extreme_child)
# print( "Children: ",len( Start.Child_States) )
# print( "counter: ", Start.counter)
# print( "counter2: ", Start.counter2)
# print( "Extreme_child children: ", len( Start.Extreme_child.Child_States) )
# print("Start: ", Start.Child_States[0])

# print( "c1", Start.Extreme_child.Child_States)
# print("Start children1: ", Start.Extreme_child)
# print("Start children2: ", Start.Extreme_child.Extreme_child)
# print("Start children3: ", Start.Extreme_child.Extreme_child.Extreme_child)
# print("Start children4: ", Start.Extreme_child.Extreme_child.Extreme_child.Extreme_child)
# print("child children1:  ")
