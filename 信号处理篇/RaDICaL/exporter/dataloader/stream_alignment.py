class SensorStreamAlignment(object):
    """Keeps a buffer of n streams, using first stream as reference.
    Callback sink is called with reference stream + other other stream with closest matching time
    stamp (before or after) when all streams have at least 1 datapoint before and after reference
    stream in buffer. 
    """
    def __init__(self, aligned_sink, streams_to_sync=3):
        super(SensorStreamAlignment, self).__init__()
        self.sink = aligned_sink
        self.streams = tuple([] for _ in range(streams_to_sync))

        for s in self.streams[1:]:
            s.append((None, 0))
        
        print("Alignment init:")
        print(list(len(s) for s in self.streams))

    def push_frame(self, data, time_stamp, streamid):
        self.streams[streamid].append((data, time_stamp))
        self.check_queues()

    def check_queues(self):
        #if group ready, call sink

        #stream 0 is used as the time reference stream
        if len(self.streams[0]) > 0:
            ref_time = self.streams[0][0][1]
    
            #check if other streams have a before and after frame
            #if before and after exist, call sink with group that is smallest time delta apart
            synced_idx = [None for _ in range(len(self.streams))]
            synced_idx[0] = 0
            for sidx, stream in enumerate(self.streams[1:]):
                before, after = None, None
                for idx, f in enumerate(stream):
                    f_time = f[1]
                    if f_time < ref_time:
                        before = idx
                    elif f_time >= ref_time:
                        after = idx
                        break

                if before is not None and after is not None:
                    t_before = ref_time - stream[before][1]
                    t_after = stream[after][1] - ref_time
    
                    if t_before <= t_after:
                        synced_idx[sidx+1] = before
                    else:
                        synced_idx[sidx+1] = after
    
            if all(s is not None for s in synced_idx):
                self.sink(tuple(s[idx] for s, idx in zip(self.streams, synced_idx)))
                for s, idx in zip(self.streams[1:], synced_idx[1:]):
                    list(s.pop(0) for _ in range(idx-1))
                self.streams[0].pop(0)

        #after calling sink, reference queue should be empty
        # other streams should only have the newest data frame

    @property
    def buffer_size(self):
        return tuple(len(s) for s in self.streams)


