#Euclidean distance
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

#inspired by blog post: https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
class Tracker():
    """Class represents object tracker, based on the centroid tracking algorithm described in the thesis."""
    def __init__(self, max_not_seen=50):
        #newly detected object is going to have this id, counting from 0
        self.next_id = 0

        #dictionary mapping unique object_id to it's bbox centroid
        self.objects = OrderedDict()

        #dictionary mapping unique object_id to the number of frames that the algorithm hasn't seen it 
        #in a case of object disappereance from the scene
        self.not_seen = OrderedDict()

        #initialize the given algorithm parameter max_not_seen
        #maximum number of consecutive frames that the previously detected object hasn't appeared in
        #after reaching the value, the algo proclaims that the object has disappeared from the scene for good
        self.max_not_seen = max_not_seen

    def register(self, centroid):
        """Registers newly detected object and assigns it new unique id."""
        self.objects[self.next_id] = centroid
        self.not_seen[self.next_id] = 0
        #update unique id
        self.next_id += 1

    def deregister(self, id):
        """Deregisters object that hasn't been present for max_not_seen number of consecutive frames."""
        del self.objects[id]
        del self.not_seen[id]

    def update(self, bboxes):
        """Handles the cetroid tracing, invoked on each frame."""

        #case when no objects has been detected on the frame
        if len(bboxes) == 0:

            #since in the current frame no objects has been detected -> update not_seen value for each
            #previously detected object
            for id in list(self.not_seen.keys()):
                self.not_seen[id] += 1

                if self.not_seen[id] > self.max_not_seen:
                    self.deregister(id)
            
            return self.objects
        
        #case when some objects has been detected in the frame
        #for each given bounding box[start_x, start_y, end_x, end_y] -> (c_x, c_y)
        new_centroids = np.zeros((len(bboxes), 2), dtype="int")
        
        for (i, (start_x, start_y, end_x, end_y)) in enumerate(bboxes):
            #use the bounding box coordinates to derive the centroid
            c_x = int((start_x + end_x) / 2.0)
            c_y = int((start_y + end_y) / 2.0)
            new_centroids[i] = (c_x, c_y)

        #when we currently aren't tracking any objects -> register them all
        if len(self.objects) == 0:
            for i in range(len(new_centroids)):
                self.register(new_centroids[i])
        
        #we are already tracking some objects
        else:
            #current traced objects
            ids = list(self.objects.keys())
            current_centroids = list(self.objects.values())

            #compute the Euclidean distance bewteen each current_centroid and each new_centroid
            #D shape -> (# current_centroids, # new_centroids)
            D = dist.cdist(np.array(current_centroids), new_centroids)
            
            #finding the minimum value of each row and by sorting these values we can obatin indexes of this values 
            #which tell us which new_centroid is closest to already exisitng object
            rows = D.min(axis=1).argsort()
            
            #similar process to the previous one, find the minimum value of each column and sort them using 
            #rows indexes
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                #we have already dealt with current object at index row or new centroid at index col
                if row in used_rows or col in used_cols:
                    continue
                
                #update the centroid
                object_id = ids[row]
                self.objects[object_id] = new_centroids[col]
                self.not_seen[object_id] = 0

                #we have examined current object at index row and new centroid at index col
                used_rows.add(row)
                used_cols.add(col) 

            #the ones we have not examined yet(lost or newly added objects)
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            
            # #of current objects >= #of newly found objects -> some object may not be present
            if D.shape[0] > D.shape[1]:
                for row in unused_rows:
                    #increment not seen value
                    object_id = ids[row]
                    self.not_seen[object_id] += 1

                    #check if reached max_non_seen
                    if self.not_seen[object_id] > self.max_not_seen:
                        self.deregister(object_id)
            # #of current objects < #of newly found objects -> new objects present
            else:
                for col in unused_cols:
                    self.register(new_centroids[col])
                for row in unused_rows:
                    #increment not seen value
                    object_id = ids[row]
                    self.not_seen[object_id] += 1

        return self.objects
            

def test_tracker():
    t = Tracker()
    t.register((0.2, 0.3))
    t.register((0.5, 0.9))

    #Registering
    assert(t.next_id == 2)
    assert(0 in t.objects)
    assert(t.objects[0] == (0.2,0.3))
    
    #Deregistering
    t.deregister(0)
    assert(t.next_id == 2)
    assert(0 not in t.objects)
    assert(t.objects[1] == (0.5,0.9))
    t.register((0.7, 0.7))
    assert(t.next_id == 3)
    
    #No detected objects for max_not_seen frames -> shoudle deregister id 1 and 2
    assert(1 in t.objects)
    assert(2 in t.objects)
    t.update([])
    assert(t.not_seen[1] == 1)
    assert(t.not_seen[2] == 1)
    for _ in range(50):
        t.update([])
    assert(1 not in t.objects)
    assert(2 not in t.objects)
    
    #No tracking of any objects and update with not empty bboxes is invoked
    t.update([(5,5,10,10), (10,10,20,20)])
    assert(t.next_id == 5)
    assert(3 in t.objects)
    assert((t.objects[3] == (7,7)).all())
    assert(4 in t.objects)
    assert((t.objects[4] == (15,15)).all())
    
    #new object found
    t.update([(5,5,10,10), (10,10,20,20), (20,15,30,20)])
    assert(t.next_id == 6)
    assert(5 in t.objects)
    assert((t.objects[5] == (25,17)).all())
    
    #first object moved a bit and its updated
    t.update([(5,5,11,11), (10,10,20,20), (20,15,30,20)])
    assert(t.next_id == 6)
    assert(3 in t.objects)
    assert((t.objects[3] == (8,8)).all())
    
    #first and third object have moved a bit and are updated
    t.update([(4,4,11,11), (10,10,20,20), (21,15,30,20)])
    assert(t.next_id == 6)
    assert(3 in t.objects)
    assert((t.objects[3] == (7,7)).all())

    assert(5 in t.objects)
    assert((t.objects[5] == (25,17)).all())
    
    #5th object has disappeared and new 6th came into frame at the same time
    t.update([(4,4,11,11), (10,10,20,20), (5000,1200,7000,1500)])
    assert(t.next_id == 7)
    assert(5 in t.objects)
    assert(6 in t.objects)
    assert((t.objects[6] == (6000,1350)).all())
    assert((t.objects[5] != (6000,1350)).all())
    
    #test if increased non_seen_param
    t.update([])
    assert(t.not_seen[6] == 1)
    assert(t.not_seen[3] == 1)
    assert(t.not_seen[4] == 1)
    assert(t.not_seen[5] == 2)

    #test non_seen_param
    t.update([(4,4,11,11), (10,10,20,20), (5000,1200,7000,1500),(0,0,1,1)])
    assert(t.not_seen[6] == 0)
    assert(t.not_seen[3] == 0)
    assert(t.not_seen[4] == 0)
    assert(t.not_seen[5] == 3)
    assert(t.not_seen[7] == 0)
    
    #test for erasing non-seen objects after 50 updates
    for _ in range(51):
        t.update([])
    assert(5 not in t.objects)
    assert(6 not in t.objects)
    assert(3 not in t.objects)
    assert(4 not in t.objects)
    assert(7 not in t.objects)

    #slight update
    t.update([(10,50,30,80)])
    assert(t.next_id == 9)
    assert((t.objects[8] == (20, 65)).all())
    t.update([(30,80,60,110)])
    assert(t.next_id == 9)
    assert((t.objects[8] == (45, 95)).all())
    
    

    return True


assert(test_tracker())
