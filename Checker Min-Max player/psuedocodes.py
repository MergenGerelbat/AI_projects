
Max_value( CurNode, level, levelthreshold):
	maxval=-inf

	for position in CurNode.White


def Max_value(CurNode, heuristic, level, level_threshold):

    maxval= -infinity

    # For each White peice, consider all 3 moves
    for x,y in CurNode.White_Positions: 
    	#Consider all three positions
    	for position in {Left-Up, Right-up, Up}:
	        x_new = position.x
	        y_new = position.y

	        if( (x_new,y_new) is legal move ):
	            ChildA= Max_Child( x_new,y_new, CurNode, heuristic, "minimax", level, level_threshold )  

	            if( ChildA.value > maxval):
	                maxval= ChildA.value
	                CurNode.Extreme_child= ChildA

    return maxval


#level_threshold is depth of the search
def Max_Child(x_new,y_new, CurNode, heuristic, searchtype, level, level_threshold, alpha, beta):

    # Create newChild (a node representing the new state of the board)
    Create new_White_Positions
    Create new_Black_Positions
    newChild = State(new_White_Positions, new_Black_Positions)

    # Calculate "Max value" of newChild
    # if search haven't reached leaf node, call use Max() to calculate "Max value"
    # If search has reached leaf node, use the heuristic function to calculate "Max value"
    # Note: "level" increases by 1 when Max() is used.
    if( level < level_threshold): 

        if( searchtype=="alphabeta"):
            newChild.value= AlphaBeta_Min_value(newChild, heuristic, level+1, level_threshold, alpha, beta)
        else if( searchtype== "minimax"):
            newChild.value= Min_value(newChild, heuristic, level+1, level_threshold)

    else if( level == level_threshold):  
    	newChild.value = heuristic( newChild )

    return newChild  


def Min_value(CurNode, heuristic, level, level_threshold):
    minval = infinity
  
    # For each Black peice, consider all 3 moves
    for x,y in CurNode.Black_Positions: 
    	#Consider all three positions
    	for position in {Left-down, Right-down, Down}
	        x_new = position.x
	        y_new = position.y

	        if( x_new, y_new is a legal move ):
	            ChildA = Min_Child(x_new,y_new, CurNode, heuristic, "minimax", level, level_threshold )

	            if( ChildA.value < minval):
	                minval= ChildA.value
	                CurNode.Extreme_child= ChildA   
    return minval






def AlphaBeta_Max_value(CurNode, heuristic, level, level_threshold):

    maxval= -infinity

    # For each White peice, consider all 3 moves
    for x,y in CurNode.White_Positions: 
    	#Consider all three positions
    	for position in {Left-Up, Right-up, Up}:
	        x_new = position.x
	        y_new = position.y

	        if( (x_new,y_new) is legal move ):
	            ChildA= Max_Child( x_new,y_new, CurNode, heuristic, "minimax", level, level_threshold )  

	            if( ChildA.value > maxval):
	                maxval= ChildA.value
	                CurNode.Extreme_child= ChildA

	            if( maxval >=beta):
	                return maxval #end the forloop. 
	            alpha= max( alpha, maxval)

    return maxval

def AlphaBeta_Min_value(CurNode, heuristic, level, level_threshold):
    minval = infinity
  
    # For each Black peice, consider all 3 moves
    for x,y in CurNode.Black_Positions: 
    	#Consider all three positions
    	for position in {Left-down, Right-down, Down}
	        x_new = position.x
	        y_new = position.y

	        if( x_new, y_new is a legal move ):
	            ChildA = Min_Child(x_new,y_new, CurNode, heuristic, "minimax", level, level_threshold )

	            if( ChildA.value < minval):
	                minval= ChildA.value
	                CurNode.Extreme_child= ChildA   

	            if( minval <=alpha):
	                return minval  
	            beta = min(beta, minval)

    return minval

def Game(searchtype_white, heuristic_white, searchtype_black, heuristic_black):
    CurNode = Initial_Node()
    player= "white"
    
    while( game_ended(CurNode) ==False ):
        if( player=="white" ):

            if( searchtype_white=="alphabeta"):
                CurNode.value = AlphaBeta_Max_value(CurNode, heuristic_white, level=0, level_threshold=4, alpha= -inf, beta = inf) 
            else:
                CurNode.value = Max_value(CurNode, heuristic_white, level=0, level_threshold=3)

            # go to next state
            CurNode = CurNode.Extreme_child
            # switch player
            player = "black"

        else if( player == "black")
            
            if(searchtype_black=="alphabeta"):
                CurNode.value = AlphaBeta_Min_value( CurNode, heuristic_black, level=0, level_threshold=4, alpha= -inf, beta = inf)
            else:
                CurNode.value = Min_value(CurNode, heuristic_white,level=0, level_threshold=3)

            # go to next state
            CurNode = CurNode.Extreme_child
            # switch player
            player = "white"
    #end of while loop

    print_game_status(CurNode)

    return

    