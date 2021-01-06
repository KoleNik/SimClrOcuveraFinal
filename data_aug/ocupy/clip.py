def frame_to_np_array(frame):
    return marshalling.asNumpyArray(frame.DepthFrame.RawData).reshape((frame.DepthFrame.Height,frame.DepthFrame.Width))

def np_array_to_displayable_np_array(np_array):
    type_change = (np_array/100).astype(np.uint8) #Might change the coloring at some point
    return np.stack([type_change,type_change,type_change], axis=2)

class Clip:
    #master_video_objects
    #current_master_video_index
    #start_frame
    #end_frame
    
    def __init__(self,s,start_frame = None,end_frame = None, mongo_connection_string = "mongodb://hv-mongo02.ng.com:27017"):
        '''
        s is an object representing either the path to an ocv or a mongoID (either as a string or an ObjectId) or patient number corresponding to a range of frames for a patient
        '''
        
        try:
            object_id = ObjectId(s)
        except:
            #TODO : try parsing as an ocv or patient number
            object_id = None
        
        #Todo: check if this is an ocv before getting the pymongo client
        
        mongo_client = pymongo.MongoClient(mongo_connection_string)
        patient_monitoring = mongo_client["PatientMonitoring"]
        
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.ocv_player = None
        self.master_videos = []
        self.master_video_index = -1
        
        self._psro = PlayerStreamReadOptions()
        self._psro.AllowMissingStreams = True
        self._psro.ReadDepth = True
        
        #The case that this is a clip
        clips = patient_monitoring["ClippedVideo"]
        clip = clips.find_one({'_id': object_id})
        if not clip is None:
            self.master_videos = self._mongo_master_videos(clip['MasterVideoIds'],mongo_client)
            self.start_frame = clip["StartFrame"]
            self.end_frame = clip["EndFrame"]

        mongo_client.close()
        
        if len(self.master_videos) == 0:
            self.master_video_index = 1
        else:
            assert(self.master_videos[0]["StartPatientFrame"] <= self.start_frame and self.master_videos[-1]["EndPatientFrame"] >= self.end_frame)
            
    def __del__(self): 
        if not self.ocv_player is None:
            self.ocv_player.Dispose()
    
    def get_frame(self,frame_number):
        camera_frame = self._get_CameraFrame(frame_number)
        return frame.FrameNumber, frame_to_np_array(frame)
        
    def next_frame(self):
        frame = self._next_CameraFrame()
        
        return frame.FrameNumber, frame_to_np_array(frame)

    def next_frame_PIL(self):
        _, frame = self.next_frame()
        displayable = np_array_to_displayable_np_array(frame)
        return Image.fromarray(displayable)
    
    def reset(self):
        self.master_video_index = -1
        self._clear_ocv_player()
    
    def moviepy_clip(self, duration = None):
        #reset
        self.reset()
        
        next_frame = None
        t0 = 0
        t1 = 0
        
        h,w = 424,512
        
        def get_frame(t):
            nonlocal next_frame, t0, t1, h, w
            first_time = (t == 0.0)

            while (t1 - t0)/1000 < t or first_time:
                next_frame = self._next_CameraFrame()
                h,w = next_frame.DepthFrame.Height, next_frame.DepthFrame.Width
                t1 = next_frame.DepthFrame.Timestamp.TotalMilliseconds
                if first_time:
                    t0 = t1
                first_time = False
                
            if next_frame is None:
                return np.zeros((h,w))
            
            return np_array_to_displayable_np_array(frame_to_np_array(next_frame))
        
        return mpy.VideoClip(make_frame = get_frame, duration=duration)
    
    def first_frame(self):
        self.reset()
        return self.next_frame()
    
    def last_frame(self):
#         frame = self._get_CameraFrame(self.end_frame)
#         if not frame is None and frame.DepthFrame.FrameNumber > self.end_frame:
#             return frame
        #If this failed, we have to go fishing for the frame
        lower = self.start_frame
        upper = self.end_frame
        while lower <= upper - 1:
            print("\r                       ",end="")
            print(f"\r{lower} {upper}",end="")
            mid = (lower + upper)//2
            frame = self._get_CameraFrameGEQ(mid)
            if frame is None:
                upper = mid-1
            else:
                lower = mid
            
        frame = self._get_CameraFrame(upper)
        if not frame is None:
            return frame
        frame = self._get_CameraFrame(lower)
        return frame
                

    def moviepy_clip_all_frame(self,fps=30):
        
        def image_generator():
            self.reset()
            
            frame = self._next_CameraFrame()
            while not frame is None:
                yield frame
                frame = self._next_CameraFrame()
            
        c = 0
        for frame in image_generator():
            c = c+1
            print(f"\rFirst pass {frame.FrameNumber}/{self.end_frame}", end="")
            
        return None
    
    def _increment_master_video(self):
        for i in range(self.master_video_index + 1,len(self.master_videos)):
            if self._set_ocv_player(i):
                return True
            
        self.master_video_index = len(self.master_videos)
        self._clear_ocv_player()
        return False
            
    def _mongo_master_videos(self,id_strings,mongo_client):
        to_return = []
        for id_string in id_strings:
            mv = mongo_client["PatientMonitoring"]["MasterVideo"].find_one({"_id" : ObjectId(id_string)})
            if mv is None:
                continue
            to_return.append(mv)
            
        return sorted(to_return,key= lambda mv : mv["StartPatientFrame"])
    
    def _set_ocv_player(self,master_video_index):
        '''
        Tries to change the current master video index and load the ocv player if necessary
        Returns true if success and False otherwise
        '''
        
        if not self.ocv_player is None and master_video_index == self.master_video_index:
            return True
        
        self._clear_ocv_player()
        
        if master_video_index > len(self.master_videos) or master_video_index < 0:
            return False
        
        self.master_video_index = master_video_index
        self.ocv_player = OCVPlayer.Open(self.master_videos[self.master_video_index]["FileLocation"])
        return True
        
        
    def _clear_ocv_player(self):
        if not self.ocv_player is None:
            self.ocv_player.Dispose()
            self.ocv_player = None
            
    def _get_CameraFrame(self, frame_number):
        i = self._index_of_master_video_containing_frame(frame_number)
        if i is None:
            return None
        
        offset = self.master_videos[i]["StartPatientFrame"] - self.master_videos[i]["VideoStartFrame"]
    
        if self._set_ocv_player(i):
            frame = self.ocv_player.ReadFrame(frame_number - offset, self._psro)
            return frame
        else:
            return None
        
    def _get_CameraFrameGEQ(self, frame_number):
        frame = self._get_CameraFrame(frame_number)
        if frame is None:
            return self._next_CameraFrame()
        else:
            return frame
        
    def _next_CameraFrame(self):
        if self.ocv_player is None:
            if self.master_video_index != -1:
                assert(self.master_video_index == len(self.master_videos))
                return None
            frame = self._get_CameraFrame(self.start_frame)
            if frame is None and self.ocv_player is not None: #This might happen if the start frame doesn't actually match a frame in the ocv 
                frame = self.ocv_player.ReadNextFrame(self._psro)
        else:
            frame = self.ocv_player.ReadNextFrame(self._psro)
            
        while frame is None:
            if not self._increment_master_video():
                return None
            frame = self.ocv_player.ReadNextFrame(self._psro)
            

        if frame.FrameNumber > self.end_frame:
            self._increment_master_video() #This will set the state we want
            return None
        
        return frame
        
    def _index_of_master_video_containing_frame(self, frame_number):
        lower = 0
        upper = len(self.master_videos) - 1
        while lower < upper:
            mid = (lower + upper) // 2
            if self.master_videos[mid]["StartPatientFrame"] > frame_number:
                upper = mid - 1
            elif self.master_videos[mid]["EndPatientFrame"] < frame_number:
                lower = mid + 1
            else: #We've found the correct master video
                lower = mid
                upper = mid
            
        if lower > upper or lower < 0 or upper >= len(self.master_videos):
            return None
        
        i = lower
        
        if self.master_videos[i]["StartPatientFrame"] <= frame_number and frame_number <= self.master_videos[i]["EndPatientFrame"]:
            return i
        
        return None
            
        