#coding:utf-8
import enginee
if __name__ == '__main__':
    #just test code.
    c = enginee.MutiTracker("enginee/videos/run.mp4",(False,True,False,False,15))
    _,ret = c.Tracker1(step=200)
    print("result:::up:%s,down:%s,left:%s,right:%s"%(ret[0],ret[1],ret[2],ret[3]))