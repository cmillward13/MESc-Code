# Writing my thesis alrogithm in Python insted of MATLAB to get around MATLAB's weird handing of complex numbers.

#Import dependencies
import pandas as pd
import numpy as np

# RTSA Calculation function
def RTSA_CALCS(path:str, side:bool, diameter:int, nsa:int):
    """
    This function caclulates the total sliding distance a humeral cup sees at 5 defined points on its articulating surface. It takes in a csv file containing rotation matrices representing the humerus' position relative to the sternum and outputs the relative motion experienced by the superior-, inferior-, anterior-, superior-, and centre-most points on the cup. It also deterines the frequency each point spends in a position medial to the glenosphere backside. Import all so you get the dependencies as well! I put the dependency calls outside of the function so we're not importing them with every call.


    inputs:
        path: path to the tsv files containing the R matrices. String type.
        side: side of arm we're analyzing. True = right, False = left. Boolean type.
        diamater: the implant diameter. Only accepts 38 or 42 mm cup diameters. Int type
        nsa: neck-shaft angle of the humeral stem. Int type.

    returns:
        distances: a list of total sliding distances (in meters) by each position.
        times: a list of the percentage of time spent in medial overlap, by position.
    """

    #Import Data
    data=pd.read_csv(path, sep="\t")


    #Set Up Implant Geometry
    r=diameter/2 #cup's inner radius
    NSA=np.radians(180-nsa) #neck-shaft angle

    #Create unit vectors representin each point on the cup during "resting" position
    if diameter==38:
        #Each point's origin after offset
        centre_o=np.array([0,1,0])
        sup_o=np.array([0,0.558947368,0.828947368]) #ISB Coordinate System
        inf_o=np.array([0,0.558947368,-0.828947368])
        ant_o=np.array([0.828947368,0.558947368,0])
        post_o=np.array([-0.828947368,0.558947368,0])

    elif diameter==42:
        #Each point's origin after offset
        centre_o=np.array([0,1,0])
        sup_o=np.array([0,0.555714286,0.831428571])
        inf_o=np.array([0,0.555714286,-0.831428571])
        ant_o=np.array([0.830952381,0.555714286,0])
        post_o=np.array([-0.830952381,0.555714286,0])

    else:
        raise ValueError('This function only supports cup diameters of either 38 mm or 42 mm. Please enter either 38 or 42')

    if side==False: #adjust cup if on the left side
        #swap all z directions
        #Change all z values
        sup_o[0]=sup_o[0]*-1
        inf_o[0]=inf_o[0]*-1
        ant_o[0]=ant_o[0]*-1
        post_o[0]=post_o[0]*-1
        centre_o[0]=centre_o[0]*-1


    #Rotate Cup Accoridng To NSA
    #Function for rotating vector
    def Rodrigues(point,axis,theta):
        """
        This function utilizes the 'Rodrigues Rotation Equation' to rotate a 3D vector about an axis.
        inputs:
            point: the point (vector) which you wish to rotate
            axis: the axis about which you wish to rotate the point
            theta: the angle (in radians) which you wish to rotate the point.
        returns:
            the vector following rotation. np.array type.
        """
        a=point*np.cos(theta) #first component
        b=np.cross(axis,point)*np.sin(theta) #second component
        c=axis*(np.dot(axis,point))*(1-np.cos(theta)) #third component
        return a+b+c

    cup_axis=np.array([1,0,0]) #rotate about the -x axis (in ISB coordinate system).
    centre_prime=Rodrigues(centre_o,cup_axis,NSA)
    sup_prime=Rodrigues(sup_o,cup_axis,NSA)
    inf_prime=Rodrigues(inf_o,cup_axis,NSA)
    ant_prime=Rodrigues(ant_o,cup_axis,NSA)
    post_prime=Rodrigues(post_o,cup_axis,NSA)
    
    #Now get R_hum wrt glenosphere
    #Step 1:Get Rhum wrt CS
    if side==True:
        R_hum_CS=np.array([data[["R00","R01","R02"]],data[["R10","R11","R12"]],data[["R20","R21","R22"]]]) #convert to np array so indexing is much faster in loop
    else:
        R_hum_CS=np.array([data[["L00","L01","L02"]],data[["L10","L11","L12"]],data[["L20","L21","L22"]]]) #convert to np array so indexing is much faster in loop
    

    #step 2: Get Rglen wrt CS
    #Get Abduction angles    
    #Step 2.1: Get Humeral Psosition
    if side==True: #if right side
        h=data[["R10","R11","R12"]].to_numpy()

    else: #if left side
        h=data[["L10","L11","L12"]].to_numpy()
    
    #Step 2.2: get abduction angles
    abd=list(map(lambda x: np.degrees(np.arccos(x.dot([0,1]))),h[:,0:2]))#in degrees to feed into matsuki curves


    
    #Step 2.3: fit lines to Matsuki curves
    #Scapular rotations from Matsuki (in MATLAB coordinate nomenclature)
    HT=[30,45,59,74,89,103,119,133] #range of humeral movement from Matsuki
    GHx=[10.3,14.5,19.2,23.9,28.9,33.2,38.9,43.3] #upward rotation
    GHz=[30.1,27.8,25.4,23.0,20.7,16.8,13.6,10.9] #forward rotation
    GHy=[23.5,23.2,22.8,22.7,22.2,21.7,18.5,18.1] #internal rotation

    #fit curve to Matsuki Points
    pX=np.polyfit(HT,GHx,1)
    pY=np.polyfit(HT,GHy,1)
    pZ=np.polyfit(HT,GHz,1)

    #Step 2.4: get rotation angles
    phi_x=-np.radians(np.polyval(pX,abd)) #threw negative integer in there because we're rotation about -x axis
    phi_y=-np.radians(np.polyval(pY,abd))
    phi_z=-np.radians(np.polyval(pZ,abd)) 

    if side==False:
        phi_x=phi_x*-1
        phi_y=phi_y*-1


    #Step 2.5: Create Rglen wrt CS   
    #Function for building R Matrix
    def Matrixify(rot_x,rot_y,rot_z):
        """
        This function builds a rotation matrix given roations about each axis
        inputs:
            rot_x: angle (in radians) to rotate about x axis
            rot_y: angle (in radians) to rotate about y axis
            rot_z: angle (in radians) to rotate about z axis
        returns:
            The corresponding R matrix as a numpy array
        """
        Rz=np.array([[1,0,0],[0,np.cos(rot_z),-np.sin(rot_z)],[0,np.sin(rot_z),np.cos(rot_z)]])
        Ry=np.array([[np.cos(rot_y),0,np.sin(rot_y)],[0,1,0],[-np.sin(rot_y),0,np.cos(rot_y)]])
        Rx=np.array([[np.cos(rot_x),-np.sin(rot_x),0],[np.sin(rot_x),np.cos(rot_x),0],[0,0,1]])
        
        return Ry@Rx@Rz
    
    R_glen_CS=[]
    for i in range(len(phi_x)):
        R_glen_CS.append(Matrixify(phi_x[i],phi_y[i],phi_z[i])) #input as ISB coordinates

    # if side==True: #getting values forglenosphere wrt CS for graphing
    #     x=np.array(list(map(lambda x: x.dot([1,0,0]),R_glen_CS)))
    #     y=np.array(list(map(lambda x: x.dot([0,1,0]),R_glen_CS)))
    #     z=np.array(list(map(lambda x: x.dot([0,0,1]),R_glen_CS)))
    # else:
    #     x=np.array(list(map(lambda x: x.dot([-1,0,0]),R_glen_CS)))
    #     y=np.array(list(map(lambda x: x.dot([0,1,0]),R_glen_CS)))
    #     z=np.array(list(map(lambda x: x.dot([0,0,1]),R_glen_CS)))


    #Step 3: Create matrix for Rhum wrt Glen
    R_hum_glen=[]
    for i in range(len(R_glen_CS)):
        R_hum_glen.append(np.linalg.inv(R_glen_CS[i])@R_hum_CS[:,i])


    #Step 4: Get position of all cup points wrt glen at each t_n
    sup=[]
    inf=[]
    ant=[]
    post=[]
    centre=[]

    for i in range(len(R_hum_glen)):
        sup.append(sup_prime.dot(R_hum_glen[i]))
        inf.append(inf_prime.dot(R_hum_glen[i]))
        ant.append(ant_prime.dot(R_hum_glen[i]))
        post.append(post_prime.dot(R_hum_glen[i]))
        centre.append(centre_prime.dot(R_hum_glen[i]))


    sup=np.array(sup)
    inf=np.array(inf)
    ant=np.array(ant)
    post=np.array(post)
    centre=np.array(centre)


    if side==False:
        #Change all z values so left and right sides look the same (primarily for graphing)
        sup[:,0]=sup[:,0]*-1
        inf[:,0]=inf[:,0]*-1
        ant[:,0]=ant[:,0]*-1
        post[:,0]=post[:,0]*-1
        centre[:,0]=centre[:,0]*-1


    # #Step 5: Check overlap
    # #Create matrix with zeros for every time it's in overlap
    # sup_overlaps=list(map(lambda x: False if x[0]<0 else True,sup))
    # inf_overlaps=list(map(lambda x: False if x[0]<0 else True,inf))
    # ant_overlaps=list(map(lambda x: False if x[0]<0 else True,ant))
    # post_overlaps=list(map(lambda x: False if x[0]<0 else True,post))
    # centre_overlaps=list(map(lambda x: False if x[0]<0 else True,centre))
    
    # #Step 6: determine change in position.
    # #Step 6.1: remove first index of overlaps .. want to have all = 0 if they end in overlapped position.
    # sup_overlaps.pop(0)
    # inf_overlaps.pop(0)
    # ant_overlaps.pop(0)
    # post_overlaps.pop(0)
    # centre_overlaps.pop(0)

    # #Step 6.2: multiply to make all deltas = 0 when overlapped, then sum and multiply by r. in meters.
    # d_sup=np.sum(np.linalg.norm(np.diff(sup,axis=0),axis=1)*sup_overlaps)*r/10**3
    # d_inf=np.sum(np.linalg.norm(np.diff(inf,axis=0),axis=1)*inf_overlaps)*r/10**3
    # d_ant=np.sum(np.linalg.norm(np.diff(ant,axis=0),axis=1)*ant_overlaps)*r/10**3
    # d_post=np.sum(np.linalg.norm(np.diff(post,axis=0),axis=1)*post_overlaps)*r/10**3
    # d_centre=np.sum(np.linalg.norm(np.diff(centre,axis=0),axis=1)*centre_overlaps)*r/10**3

    # #Step 7: Determine Overlap Time
    # t_overlap_sup=1-(sum(sup_overlaps)/len(sup_overlaps))
    # t_overlap_inf=1-(sum(inf_overlaps)/len(inf_overlaps))
    # t_overlap_ant=1-(sum(ant_overlaps)/len(ant_overlaps))
    # t_overlap_post=1-(sum(post_overlaps)/len(post_overlaps))
    # t_overlap_centre=1-(sum(centre_overlaps)/len(centre_overlaps))
    
    # #Step 8: Export variables
    # distances=[d_sup, d_inf, d_ant, d_post, d_centre]
    # times=[t_overlap_sup, t_overlap_inf, t_overlap_ant, t_overlap_post, t_overlap_centre]
    # fname=path.split("/")[-1]
    # return(fname,nsa,diameter,distances, times)



    #Debugging
    vals=inf[0:100,0]
    cutoff=0.01
    print(np.where(np.logical_and(vals>-cutoff,vals<cutoff)))
   
    print(h[72])
    # import matplotlib.pyplot as plt
    # fig=plt.plot()
    # ax=plt.axes(projection='3d')
    # # ax.view_init(azim=-93,elev=124)
    # # ax.view_init(azim=0,elev=0)
    # ax.view_init(azim=-90,elev=90) #Z-Y view
    # # ax.view_init(azim=-90,elev=0) #Z-X view
    # top=100
    # step=1
    # for i in range(0,top,step):
    #     plt.cla()
    #     ax.set_xlim(-1,1)
    #     ax.set_ylim(-1,1)
    #     ax.set_zlim(-1,1)
    #     ax.set_xlabel('Z')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('X')
    #     plt.plot([0,h[i,0]],[0,-h[i,1]],[0,h[i,2]],color='black')
    #     # plt.plot([0,inf_CS[i][0]],[0,-inf_CS[i][1]],[0,inf_CS[i][2]],color='green')
    #     plt.plot([0,inf[i,0]],[0,-inf[i,1]],[0,inf[i,2]],color='green')
    #     # plt.plot([0,sup[i,0]],[0,-sup[i,1]],[0,sup[i,2]],color='pink')
    #     # plt.plot([0,ant[i,0]],[0,-ant[i,1]],[0,ant[i,2]],color='blue')
    #     # plt.plot([0,post[i,0]],[0,-post[i,1]],[0,post[i,2]],color='red')  
    #     plt.plot([0,centre[i,0]],[0,-centre[i,1]],[0,centre[i,2]],color='yellow')        
    #     # plt.plot([0,inf_prime[2]],[0,-inf_prime[1]],[0,inf_prime[0]],color='green')
    #     # plt.plot([0,sup_prime[2]],[0,-sup_prime[1]],[0,sup_prime[0]],color='red') 
    #     # plt.plot([0,ant[i,0]],[0,-ant[i,1]],[0,ant[i,2]],color='blue')
    #     # plt.plot([0,x[i,0]],[0,-x[i,1]],[0,x[i,2]],color='yellow')
    #     # plt.plot([0,y[i,0]],[0,-y[i,1]],[0,y[i,2]],color='blue')
    #     # plt.plot([0,z[i,0]],[0,-z[i,1]],[0,z[i,2]],color='cyan') 
    #     plt.pause(0.01)

