import os,glob,sys,shutil
import numpy
from rasotools.anomaly import *
from Magics.macro import *
import netCDF4
from rasotools.utils import *
from rasotools.dailyseries import *
import time
import copy
from collections import OrderedDict
#import matplotlib.pyplot as plt
#from matplotlib.patches import Polygon
from rasotools.define_datasets import *
import h5py
import pandas as pd

def add_meanandspread(fn):
    try:
        with netCDF4.Dataset(fn,'r') as src, netCDF4.Dataset(fn[:-3]+'ens.nc','w') as dst:
        
            for name, dimension in list(src.dimensions.items()):
                dst.createDimension(name, len(dimension) if not dimension.isunlimited() else None)
        
            spdict={'ce20cens_mean':numpy.zeros(src.variables['ce20c0_andep'].shape,'float32'),
                        'ce20cens_spread':numpy.zeros(src.variables['ce20c0_andep'].shape,'float32')}
            for name, variable in list(src.variables.items()):
        
                # take out the variable you don't want
                if name == 'some_variable': 
                    continue
        
                
                if '_FillValue' in variable.ncattrs():
                    x = dst.createVariable(name, variable.datatype, variable.dimensions,fill_value=getattr(variable,'_FillValue'))
                else:
                    x = dst.createVariable(name, variable.datatype, variable.dimensions)
                    
                dst.variables[name][:] = src.variables[name][:]
    
                print((name,variable.ncattrs()))
                j=0
                for j in variable.ncattrs():
                    o=getattr(variable,j)
                    if type(o)==str:
                        o=str(o)
                    if j!='_FillValue' and j!='scale_factor' and j!='add_offset':
                        setattr(dst.variables[name],j,o)
                        
                if 'ens' in name:
                    spdict['ce20cens_mean'][:]=0.
                    spdict['ce20cens_spread'][:]=0.
                    setattr(dst.variables[name],'units','K')
                    
                    
            
            for i in range(10):
                ename='ce20c{}_andep'.format(i)
                spdict['ce20cens_mean'][:]+=src.variables[ename][:]
                spdict['ce20cens_spread'][:]+=src.variables[ename][:]*src.variables[ename][:]
            spdict['ce20cens_mean']=spdict['ce20cens_mean']/10.
            spdict['ce20cens_spread']=numpy.sqrt(spdict['ce20cens_spread']/10.-spdict['ce20cens_mean']*spdict['ce20cens_mean']+0.000001)
            spdict['ce20cens_mean'][spdict['ce20cens_mean']>1.e10]=numpy.nan
            spdict['ce20cens_spread'][spdict['ce20cens_mean']>1.e10]=numpy.nan
            for name in list(spdict.keys()):
                try:
                    x = dst.createVariable(name, src.variables['ce20c0_andep'].datatype, src.variables['ce20c0_andep'].dimensions,
                                           fill_value=getattr(src.variables['ce20c0_andep'],'_FillValue'))
                except:
                    print((fn+': '+name+' already exists'))
                dst.variables[name][:] = spdict[name][:]
                
                setattr(dst.variables[name],'long_name',name)
                setattr(dst.variables[name],'units','K')
                        
            for i in src.ncattrs():
                o=getattr(src,i)
                if type(o)==str:
                    o=str(o)
                setattr(dst,i,o)
                
        shutil.copy(fn[:-3]+'ens.nc',fn)
    except:
        print((fn,' could not be processed'))
        

def readscalar(d,tasks,fn,dc,istat,statid,ens,pindex,minlen,lats=None,lons=None,ps=None,stlongnames=None):
    print((d["shortname"]+' '+fn))
    found=False
    if os.path.isfile(fn):


        #if fnold!=fn[ifile]:
            #try:
                #f.close()
            #except:
                #pass

        with netCDF4.Dataset(fn,"r") as f:

            #fnold=fn[ifile]

            try:
                fp = f.variables['press']
                ps=fp[:]
                fpu = fp.getncattr('units')
                if fpu=='Pa' or fpu == '':
                    ps=numpy.array(ps)/100
            except:
                return False,0,ps            
            try:
                timeoffset=daysbetween(d["startdate"],f.variables["datum"].units)        
            except:
                return False,0,ps
            try:
                dat=np.array(f.variables['datum'][0,:],dtype=numpy.int64)
            except ValueError:
                dat=np.array(f.variables['datum'][:],dtype=numpy.int64)
            
            if 'seconds' in f.variables['datum'].getncattr('units') or dat[-1] > 50000:
                dat=dat//86400

            nc_miss_val=numpy.float32(-999.)
    
            if dc==0:
                try:
                    if f.variables['lat'].size>1:
                        lats[istat]=f.variables['lat'][:].flatten()[-1]
                        lons[istat]=f.variables['lon'][:].flatten()[-1]
                    else:
                        lats[istat]=f.variables['lat'][:]
                        lons[istat]=f.variables['lon'][:]
                    if 'units' in f.variables[d['dvar']].ncattrs():
                        d['units']=f.variables[d['dvar']].getncattr('units')
                    if 'Stationn' in f.ncattrs()[-1]:
                        stlongnames.append(f.getncattr(f.ncattrs()[-1]))
                    elif 'unique_source_identifier' in f.ncattrs():
                        stlongnames.append(f.getncattr('unique_source_identifier'))
                    else:
                        stlongnames.append('')
                    dc=1
                except:
                    if len(stlongnames)<istat+1:
                        stlongnames.append('')
                    pass
    
            if type(d["dindex"]) is list:
                if dat.shape[0]<minlen:
                    toofew=True
                    print((statid+' series too short'))
                    d["ddata"][istat,:,:,:,:].fill(numpy.nan)
                    return False,0,ps
                found=True
                index=dat-1+timeoffset
            else:
                index=dat-1+timeoffset
                d["dindex"][istat,ens,0:len(index)]=index
                index=numpy.arange(len(index),dtype=int)   
    
            if d['dvar'] in list(f.variables.keys()):
                try:
                    nc_miss_val=f.variables[d['dvar']].getncattr('_FillValue')
                except:
                    pass
                try:
                    nc_miss_val=f.variables[d['dvar']].getncattr('missing_value')
                except:
                    pass
                try:
                    copystride(d["ddata"],numpy.ma.filled(f.variables[d['dvar']][:],numpy.nan),numpy.ma.filled(index),istat,ens,pindex,nc_miss_val)
                    if numpy.nanmax(d["ddata"][istat,:,:,:])>900 or numpy.nanmin(d["ddata"][istat,:,:,:])<-900:
                        print(('spurious values in ',fn))
        #                                os.remove(fn)
                    else:
                        found=True
                    if d['shortname']=='obs':
                        d['maxindex'][istat]=index[-1]
                        d['minindex'][istat]=index[0]
                    else:
                        d['maxindex'][istat]=tasks[0]['maxindex'][istat]
                        d['minindex'][istat]=tasks[0]['minindex'][istat]
                    try:
                        if d["name"]==d["shortname"]:
                            d["name"]=getattr(f.variables[d['dvar']],'long_name')
                    except:
                        pass
                except IOError:
                    print((d['shortname']+' not found'))
                    d["ddata"][istat,ens,:,:,:].fill(numpy.nan)
            else:
                print((fn,':',d['dvar'], 'not found')) 
            
    return found,dc,ps

def readvector(d,tasks,fn,dc,istat,statid,ens,pindex,minlen,lats=None,lons=None,ps=None,stlongnames=None):
    uobs=copy.deepcopy(d)
    uobs['dvar']='uwind'
    uobs['shortname']='uwind'
    vobs=copy.deepcopy(d)
    vobs['shortname']='vwind'
    vobs['dvar']='vwind'
    udep=copy.deepcopy(d)
    vdep=copy.deepcopy(d)
    found=False
    found,dc,ps=readscalar(uobs,tasks,fn[0],dc,istat,statid,ens,pindex,minlen,lats,lons,ps,stlongnames)
    found,dc,ps=readscalar(vobs,tasks,fn[1],dc,istat,statid,ens,pindex,minlen,lats,lons,ps,stlongnames)
#    for i in range(4):
#        dummy=stlongnames.pop()
    if d['shortname']!='obs':
        found,dc,ps=readscalar(udep,tasks,fn[0],dc,istat,statid,ens,pindex,minlen,lats,lons,ps,stlongnames)
        found,dc,ps=readscalar(vdep,tasks,fn[1],dc,istat,statid,ens,pindex,minlen,lats,lons,ps,stlongnames)
        uref=uobs['ddata']+udep['ddata']
        vref=vobs['ddata']+vdep['ddata']
    if d['parameter']=='wspeed':
        wsobs=numpy.sqrt(uobs['ddata']*uobs['ddata']+vobs['ddata']*vobs['ddata'])
        if d['shortname']=='obs':
            d['ddata']=copy.copy(wsobs)
        else:
            wsref=numpy.sqrt(uref*uref+vref*vref)
            d['ddata']=copy.copy(wsobs-wsref)
        d['units']='m/s'
    else:
        wdobs=270.-180./numpy.pi*numpy.arctan2(uobs['ddata'],vobs['ddata'])    
        wdobs[wdobs>360.]-=360.
        if d['shortname']=='obs':
            d['ddata']=copy.copy(wdobs)
        else:
            wdref=270.-180./numpy.pi*numpy.arctan2(uref,vref) 
            wdref[wdref>360.]-=360.
            wddiff=wdobs-wdref
            wddiff[wddiff>180.]-=360.
            wddiff[wddiff<-180.]+=360.
            d['ddata']=copy.copy(wddiff)
        d['units']='deg'
    return found,dc,ps

def mon_mean(data,days):
    sdays=pd.date_range(start='1900-01-01',end='2023-02-01',freq='MS')
    idays=np.array([(sdays[i]-sdays[0]).days for i in range(len(sdays))])
    montemp=[]
    good=[]
    gdays=[]
    for i in range(len(idays)-1):
        start,stop=np.searchsorted(days,idays[i:i+2]+1)
        if stop>start+1:
            d=data[:,:,start:stop]
            x=np.sum(~np.isnan(d),axis=2).reshape((data.shape[0],data.shape[1],1))
            if np.sum(x)>0:
                
                good.append(x.reshape((data.shape[0],data.shape[1],1)))
                montemp.append(np.nanmean(d,axis=2).reshape((data.shape[0],data.shape[1],1)))
                gdays.append(idays[i])
    
    return np.concatenate(montemp,axis=2),np.concatenate(good,axis=2),np.array(gdays)+1

def readsuny(d,tasks,fn,dc,istat,statid,ens,pindex,minlen,lats=None,lons=None,ps=None,stlongnames=None):
    try:
        spress=np.array((10.,20,30,50,70,100,150,200,250,300,400,500,700,850,925,1000))
        with h5py.File(fn) as h:
            station_name=''
            df_press=h['pressure'][:]
            idx=np.searchsorted(df_press,spress*100)
            idx[idx==df_press.shape[0]]=df_press.shape[0]-1
            
            refdate=datetime.datetime(1900,1,1)
            hty=h['time'][:]//10000
            htm=(h['time'][:]%10000)//100
            htd=h['time'][:]%100
            df_time=[datetime.datetime(hty[i],htm[i],htd[i]) for i in range(hty.shape[0])]
            df_days=np.array([(df_time[i]-refdate).days for i in range(hty.shape[0])])+1
            #df_time=pd.to_datetime(df_days-1,unit='d',origin='19000101').values
            
            mask=h['rawT'][:]==-9999.
            x=(h['rawT'][:]-h['homoT'][:])*0.1
            x[mask]=np.nan
            x=np.einsum('kli->ilk', x)
            for i in range(len(idx)):
                if not any(df_press==spress[i]*100):
                    x[:,i,:]=np.nan
            
            df_press=spress
            x[x>400]=np.nan
            
            for ih in 0,1:
                
                d['ddata'][0,0,ih,0,df_days]=x[ih,pindex,:]
            
            stlongnames.append(fn.split('/')[-1][:-3])
            
                
    except:
        return False,0,spress
        #with h5py.File(tup[0]) as o:
            
            #df_lat=o['lat'][:]
            #df_lon=o['lon'][:]
        #mask=np.zeros((2,3,df_time.shape[0]),dtype=bool)
    return True,1,spress
    
def read_dailyseries(path,tasks,nml,stnames,pindex,minlen=3600,ref=''):

#    presats=open('presatstations.t','w')
    stlongnames=[]
    str=[' ']
    fnold=''

    lats=numpy.empty(tasks[0]["ddata"].shape[0],numpy.float32)
    lons=lats.copy()
    ps=numpy.empty(16)
    stlongnames=[]
    istat=0
    goodsts=[]
    c=0
    for statid in stnames: #[0:30]:
        found=False
        toofew=False

        dc=0
        for d in tasks:
#            d["ddata"][istat,:,:,:,:].fill(numpy.nan)

            for ens in d["ens"]:
                if "dfile" in d:
                    prefix=d["dfile"][:]
                else:
                    prefix=d["file"][:]

                if "prefix" in d:
                    relpath=ref+d["prefix"][0]
                else:
                    relpath=ref+"./"
                    
                if "relpath" in d:
                    relpath=d["relpath"][0]
                else:
                    relpath=ref+"./"
                
                if "dsuff" in d:
                    suffix=d["dsuff"][:]
                else:
                    suffix=['']

#                    nanfill=True
                fn=list()
                for ifile in range(len(prefix)):
                    
                    if len(suffix)>len(prefix): # wind series
                        fn=fn+[os.path.join(path,relpath,statid,prefix[ifile]+statid+suffix[ifile]+".nc"),
                               os.path.join(path,relpath,statid,prefix[ifile]+statid+suffix[2*ifile+1]+".nc")]
                        
                        found,dc,ps=readvector(d,tasks,fn,dc,istat,statid,ens,pindex,minlen,lats,lons,ps,stlongnames)
                    elif 'SUNY' in prefix[0]:
                        fn=glob.glob(os.path.join(path,relpath,statid,prefix[ifile]+'homo-raw-subdaily-station/*'+statid+".nc"))
                        found,dc,ps=readsuny(d,tasks,fn[ifile],dc,istat,statid,ens,pindex,minlen,lats,lons,ps,stlongnames)
                        
                    else:
                        if len(d["ens"])==1:
                            fn=fn+[os.path.join(path,relpath,statid,prefix[ifile]+statid+suffix[ifile]+".nc")]
                        else:
                            fn=fn+[os.path.join(path,relpath,statid,prefix[ifile]+"{0:0>2}".format(ens)+'_'+statid+suffix[ifile]+".nc")]


                        found,dc,ps=readscalar(d,tasks,fn[ifile],dc,istat,statid,ens,pindex,minlen,lats,lons,ps,stlongnames)
                        if '-' in d['shortname']:
                            found=True

        if found and not toofew:
            goodsts.append(statid)
            print(statid)      
            istat+=1

        c+=1

    if istat<len(stnames):
        for k in range(len(tasks)):
            tasks[k]["ddata"]=tasks[k]["ddata"][0:istat,:,:,:,:]
        try:
            lats=lats[0:istat]
            lons=lons[0:istat]
            stlongnames=stlongnames[0:istat]
        except:
            lats=[]
            lons=[]
            ps=[]
            print('No data found')
#	    sys.exit(0)


    return istat,lats,lons,ps,goodsts,stlongnames

def statscatter(path,tasks,plotproperties,nml,stnames,minlen=360,ref=''):

    tt=time.time()
    print(('I am in '+path+ ' should be in ',plotproperties["exps"]))
    pindex=plotproperties["pindex"]
    try:
        dummy=plotproperties['dummy']
    except:
        dummy=''
    try:
        with open(path+'stnames','r') as f:
            stnames=f.read().split('\n')
            if len(stnames[-1])==0:
                stnames.pop()
    except:
        stnames=glob.glob(os.path.expandvars("$FSCRATCH/ei6/[0-9][0-9]????/feedbackmerged0?????.nc"))
        for k in range(len(stnames)-1,-1,-1):
            try:
                with netCDF4.Dataset(stnames[k],'r') as f:
                    if numpy.where(f.variables['datum'][0,:]>39000)[0].shape[0]>100:
                        stnames[k]=stnames[k].split('/')[-2]
                    else:
                        del stnames[k]
            except:
                    del stnames[k]
        with open(path+'stnames','w') as f:
            f.write('\n'.join(stnames))
       
#    return read_dailyseries(path,tasks,stnames,pindex,minlen=minlen)
    for l in range(len(tasks)):
        tasks[l]['ddata']=numpy.empty((len(stnames),1,2,tasks[l]['ddata'].shape[3],tasks[l]['ddata'].shape[4]))
        if type(tasks[l]['dindex']) is numpy.ndarray:
            tasks[l]['dindex']=numpy.empty((len(stnames),1,tasks[l]['dindex'].shape[2]),dtype=numpy.int32)
            tasks[l]['dindex'][:]=0
    istat,lats,lons,ps,goodsts,stlongnames=read_dailyseries(path,tasks,nml,stnames,pindex,minlen=minlen)
    
    if isinstance(goodsts,list):
        goodsts=numpy.asarray(goodsts)
    istnames=goodsts.astype(numpy.int32)
    tmshape=tasks[0]["ddata"].shape

    data=OrderedDict()
    for d in tasks:
        ens=0
        # replace RAOBCORE adjustments with unadjusted series
        if d["shortname"] in ("era5v2","era5v3","eraibc","operbc","erai_fggpsdep","erai_fggpswetdep","eijra_fgdep"):  
            data[d["shortname"]]=d["ddata"][:,ens,:,:,:]
        elif d["shortname"] in ("bgdep","bgpresat","bgdepe5"):  
            data[d["shortname"]]=-d["ddata"][:,ens,:,:,:]
        elif d["shortname"] in ["rcorr"] or 'rio' in d["shortname"][0:3] or 'rit' in d["shortname"][0:3]:
            for ens in d["ens"]:
                if ens==0:
                    dds=d["ddata"].shape
                    nullref=numpy.zeros((dds[0],dds[2],dds[3],nml['rfpar']['nmax']))
                    data[d["shortname"]]=copy.copy(nullref)
#                    data['current']=copy.copy(nullref)
#                    data['accum']=copy.copy(nullref)
                expandandadd(d["ddata"],nullref,d["dindex"],numpy.arange(nullref.shape[2],dtype=numpy.int64),
                             ens,data[d["shortname"]],-1.0)
                pass
#                data['current'][:]=data['accum'][:]
#            data[d["shortname"]]=data['current'].copy()
        print((time.time()-tt))
    
    data["gpscorr"]=data["eijra_fgdep"]-data["erai_fggpsdep"]
    for istat in range(data["gpscorr"].shape[0]):
        for ipar in range(2):
            for ip in range(data["gpscorr"].shape[2]):
                mask=~numpy.isnan(data["gpscorr"][istat,ipar,ip,:])
                if numpy.sum(mask)<180:
                    data["gpscorr"][istat,ipar,ip,:]=numpy.nan
    mask=numpy.isnan(data["gpscorr"])
    print(('setnan',time.time()-tt))
    for dd in list(data.keys())+['rcorr']:
        data[dd][mask]=numpy.nan
    tidx=calcdays(19000101,1404)
    paralist=['era5v2','eijra_fgdep','operbc']
    for ip in range(pindex.shape[0]):
        cc=dict()
        cm=dict()
        plot=False
        data['operbc']=-data['operbc']
        for p in paralist:
            cc[p]=numpy.empty(len(goodsts))
            cm[p]=numpy.empty((2,len(goodsts)))
        cm['gpscorr']=numpy.empty((2,len(goodsts)))
        cm['rcorr']=numpy.empty((2,len(goodsts)))
        cc['rcorr']=numpy.empty((len(goodsts)))
        for ifig in range(data['gpscorr'].shape[0]):
            gpsc=monmean(data["gpscorr"][ifig,:,ip,:],tidx)
            if plot: 
                f=plt.figure(figsize=(8,5))
            l=0
            for p in paralist+['rcorr']:
                l+=1
                if plot: 
                    plt.subplot(2,2,l)
                    plt.plot(gpsc.flatten(),monmean(data[p][ifig,:,ip,:],tidx).flatten(),'ko')
                mask=~numpy.isnan(gpsc)#+monmean(data[p][ifig,:,ip,:],tidx))
                if numpy.sum(mask)>10:
                    cc[p][ifig]=numpy.corrcoef(gpsc[mask],monmean(data[p][ifig,:,ip,:],tidx)[mask])[0,1]
                    cm[p][:,ifig]=numpy.nanmean(monmean(data[p][ifig,:,ip,:],tidx),axis=1)
                    if p!='rcorr' and any(numpy.abs(cm[p][:,ifig])>4):
                        print((goodsts[ifig],p,cm[p][:,ifig]))
                        cm[p][:,ifig]=numpy.nan
                else:
                    cc[p][ifig]=numpy.nan
                    cm[p][:,ifig]=numpy.nan
                if plot: 
                    plt.title(p+' vs gpscorr {:4.2f}'.format(cc[p][ifig]))
            cm['gpscorr'][:,ifig]=numpy.nanmean(gpsc,axis=1)
            if plot: 
                plt.subplot(2,2,4)
                plt.title(goodsts[ifig])
                plt.tight_layout()
                plt.savefig(path+'gpscorr_'+goodsts[ifig]+'_{:1d}'.format(int(ps[pindex[ip]]))+'.eps')
                plt.close()
            #print path+'gpscorr_'+goodsts[ifig]+'.eps {:5.2f}'.format(time.time()-tt)
            
            
        for p in paralist:
            plt.plot(cc[p],label=p+' {:3.2f}'.format(numpy.nanmean(cc[p])))
        plt.legend()
        plt.title('Correlations')
        plt.savefig(path+'allgpscorr_{:1d}'.format(int(ps[pindex[ip]]))+'.eps')
        plt.close()
        
        for p in paralist+['rcorr']:
            plt.plot(cm[p][0,:],label=p+' 00 {:3.2f}'.format(numpy.nanstd(cm[p][0,:])))
            plt.plot(cm[p][1,:],label=p+' 12 {:3.2f}'.format(numpy.nanstd(cm[p][1,:])))
        plt.legend()
        plt.title('Mean Deviation {}hPa'.format(int(ps[pindex[ip]])))
        plt.savefig(path+'allgpsmean_{:1d}'.format(int(ps[pindex[ip]]))+'.eps')
        plt.close()
        
        for p in paralist+['rcorr']:
            for ipar in range(2):
#                mask=~numpy.isnan(cm['gpscorr'][ipar,:]+cm[p][ipar,:])
#                perca=numpy.percentile(cm['gpscorr'][ipar,:],numpy.arange(100))
#                percb=numpy.percentile(cm['gpscorr'][ipar,:])cm[p][ipar,:],numpy.arange(100))
                plt.plot(numpy.sort(cm['gpscorr'][ipar,:]),numpy.sort(cm[p][ipar,:]),
                        label=p+' {:0>2} {:3.2f}'.format(ipar*12,numpy.nanstd(cm[p][ipar,:])))
        plt.legend(loc=2)
        rr=2
        plt.plot([-rr,rr],[-rr,rr])
        plt.xlim([-rr,rr])
        plt.ylim([-rr,rr])
        plt.title('QQ {}hPa'.format(int(ps[pindex[ip]])))
        plt.savefig(path+'allgpsqq_{:1d}'.format(int(ps[pindex[ip]]))+'.eps')
        plt.close()
        for p in paralist+['rcorr']:
            mask=~numpy.isnan(cm['gpscorr'][0,:]+cm[p][0,:])
            plt.plot(cm['gpscorr'][0,:],cm[p][0,:],'o',label=p+' 00 {:3.2f}'.format(numpy.corrcoef(cm['gpscorr'][0,mask],cm[p][0,mask])[0,1]))
            plt.plot(cm['gpscorr'][1,:],cm[p][1,:],'o',label=p+' 12 {:3.2f}'.format(numpy.corrcoef(cm['gpscorr'][1,mask],cm[p][1,mask])[0,1]))
        plt.legend(loc=4)
        plt.xlabel('gpscorr')
        plt.title('adjustment vs gps deviation {:1d}hPa'.format(int(ps[pindex[ip]])))
        plt.savefig(path+'allgpsmeanscatter_{:1d}'.format(int(ps[pindex[ip]]))+'.eps')
        plt.close()
    
    return

def statanalysis(path,tasks,plotproperties,nml,stnames,minlen=360,ref=''):

    tt=time.time()
    print(('I am in '+path+ ' should be in ',plotproperties["exps"]))
    pindex=plotproperties["pindex"]
    try:
        dummy=plotproperties['dummy']
    except:
        dummy=''
       
#    return read_dailyseries(path,tasks,stnames,pindex,minlen=minlen)
    istat,lats,lons,ps,goodsts,stlongnames=read_dailyseries(path,tasks,nml,stnames,pindex,minlen=minlen)

    if istat==0:
        return
    diff=''
    if ref != '':
        tasks2=copy.deepcopy(tasks)
        istat2,lats2,lons2,ps2,goodsts2=read_dailyseries(path,tasks2,nml,stnames,pindex,minlen=minlen,ref=ref)
        diff='_diff'

        for j in range(len(tasks)):
            tasks[j]['ddata']-=tasks2[j]['ddata']

#    sys.exit()
    if isinstance(goodsts,list):
        goodsts=numpy.asarray(goodsts)
#    istnames=goodsts.astype(numpy.int32)
    istnames=goodsts.astype(numpy.dtype('|S6'))
    tmshape=tasks[0]["ddata"].shape

    andeplist,andepstdlist,andeprmslist=make_andeplist()

    data=OrderedDict()
    for d in tasks:
        ens=0
        # replace RAOBCORE adjustments with unadjusted series
        t=time.time()
        if d["shortname"] in ("tm",'obs',"an20cr",'rad','rharm_h','radrharm_h','rharm','radrharm'):
            if plotproperties['parameters'][0]=='temperatures':
                #data['current']=dailyanomalydriver(d,ens,plotproperties['intervals'])
                data['current']=d["ddata"][:,ens,:,:,:]
                for ip in range(data['current'].shape[2]):
                    data['current'][:,:,ip,:]-=numpy.nanmean(data['current'][:,:,ip,:])
                    
            else:
                data['current']=d["ddata"][:,ens,:,:,:]
            if d["shortname"] in ('rad','radrharm_h','radrharm'):
                data['current'][:,0,:,:]=data['current'][:,1,:,:]-data['current'][:,0,:,:]
                data['current'][0,1,:,:]=numpy.nan
            else:
                data[d["shortname"]]=data['current'].copy()                
        elif d["shortname"] in ["tmcorr","radc"]:
            data['current']=d["ddata"][:,ens,:,:,:]-data['current']
            #if ens==0:
                #expandandadd(d["ddata"],data['current'],d["dindex"],
                             #numpy.arange(data['current'].shape[2],dtype=numpy.int64),ens,data['current'],-1.0)

            if d["shortname"] in ('radc'):
                data['current'][:,0,:,:]=data['current'][:,1,:,:]-data['current'][:,0,:,:]
                data['current'][0,1,:,:]=numpy.nan
            else:
                data[d["shortname"]]=data['current'].copy()                
        elif d["shortname"] in ['bgpresat']:  
            data["current"]=d["ddata"][:,ens,:,:,:]
            data[d["shortname"]]=data['current'].copy()
        elif d["shortname"] in ['sunycorr', 'rharmbc']:  
            data["current"]=d["ddata"][:,ens,:,:,:]
            data[d["shortname"]]=data['current'].copy()
        elif d["shortname"] in ['e20c_pandep']:  
            data["current"]=d["ddata"][:,ens,:,:,:]
            data[d["shortname"]]=data['current'].copy()
        elif d["shortname"] in ['era5v2','era5v2429','era5v3','era5v4','era5v5','era5v7','era5v2rich','era5v4rich','era5v5rich','era5v7rich','eraibc']:  
            fak=1.0
            if d["shortname"] in ["era5v2429"]:
                fak=-1.0
            data["current"]=-fak*d["ddata"][:,ens,:,:,:]
            data[d["shortname"]]=data['current'].copy()
        elif d["shortname"]=='bgdep':  
            if plotproperties['fgdepname']=='fg_dep':
                data["current"]=d["ddata"][:,ens,:,:,:]
            else:
                data["current"]=-d["ddata"][:,ens,:,:,:]
                
            data[d["shortname"]]=data['current'].copy()
        elif d["shortname"] in andeplist and d["shortname"]!='bgdep':  
            data["current"]=d["ddata"][:,ens,:,:,:]
            data[d["shortname"]]=data['current'].copy()
        elif d["shortname"] == "jra55":  
            data["current"]=data['current']+data['jra55_andep']
        elif d["shortname"] in ("era5v4-rcorr","era5v5-rcorr"):
            data["current"]=data[d["shortname"].split('-rcorr')[0]]-data['rcorr']              
        elif 'bgdep-' in d["shortname"]:
            data["current"]=data['bgdep']-data[d["shortname"].split('bgdep-')[1]]            
        elif d["shortname"] in ("20cr-e20c","presat-e20c"):
            data["current"]=dailyanomalydriver(d,ens,plotproperties)-data['e20c']              
        elif d["shortname"] == "aninc":
            data["current"]=d["ddata"][:,ens,:,:,:]-data['bgdep']
        elif d["shortname"] in ("e20c-ce20c"):
            data["current"]=d["ddata"][:,ens,:,:,:]-data['ce20c']
        elif d["shortname"] in ["rcorr"] or 'rio' in d["shortname"][0:3] or 'rit' in d["shortname"][0:3]:
            
            if len(d['ens'])>1:
                for ens in d["ens"]:
                    if ens==0:
                        dds=d["ddata"].shape
                        nullref=numpy.zeros((dds[0],dds[2],dds[3],nml['rfpar']['nmax']),dtype=numpy.float32)
                        data['current']=copy.copy(nullref)
                        data['accum']=copy.copy(nullref)
                    expandandadd(d["ddata"],data['current'],d["dindex"],numpy.arange(nullref.shape[2],dtype=numpy.int64),
                                 ens,data['accum'],1.0)
                    data['current'][:]=data['accum'][:]
                try:
                    data['current']/=numpy.array(d["ens"]).size
                except:
                    pass
            else:
                    dds=d["ddata"].shape
                    nullref=numpy.zeros((dds[0],dds[2],dds[3],nml['rfpar']['nmax']),dtype=numpy.float32)
                    data['current']=numpy.zeros((dds[0],dds[2],dds[3],nml['rfpar']['nmax']),dtype=numpy.float32)
                    expandandadd(d["ddata"],nullref,d["dindex"],numpy.arange(nullref.shape[2],dtype=numpy.int64),
                                 ens,data['current'],1.0)
                    data[d['shortname']]=numpy.empty_like(data['current'])
                    data[d['shortname']][:]=data['current']
               
            print((d['shortname'],d["ddata"][0,0,0,d['ddata'].shape[3]-1,:]))
               

        elif 'bgdepr' in d['shortname']:
            for bgc in ["bgdeprio","bgdeprit","bgdeprcorr"]: 
                if bgc in d['shortname']:#,"eijradeprcorr","eijradepriocorr","eijradepritcorr"]:
                    expandandadd(d["ddata"],data['bgdep'],d["dindex"],
                         numpy.arange(data['bgdep'].shape[2],
                                      dtype=numpy.int64),ens,data['current'],-1.0)
        elif 'obsr' in d['shortname']:
            for bgc in ["obsrio","obsrit","obsrcorr"]: 
                if bgc in d['shortname']:#,"eijradeprcorr","eijradepriocorr","eijradepritcorr"]:
                    expandandadd(d["ddata"],data['obs'],d["dindex"],
                         numpy.arange(data['obs'].shape[2],
                                      dtype=numpy.int64),ens,data['current'],-1.0)
        else:
            print((d["shortname"],' not implemented'))

#            print 'expand',ens,time.time()-t
#            plotlist=list()
        spage=page(super_page_y_length=pindex.shape[0]*plotproperties['dailyseriesheight']+0.5,
           super_page_x_length=21.0)
        if istnames.shape[0]==1 or not plotproperties["super"]:
            for istat in range(istnames.shape[0]): #[0:30]:
                mktmppath(plotproperties['tmppath']+'/'+goodsts[istat])
                for ipar in range(2):
                    if '{:0>2}'.format(ipar*12) in plotproperties['time']:

    #Setting the output
                        oname=plotproperties['tmppath']+'/'+goodsts[istat]+'/'+goodsts[istat]+'_'+d["shortname"]+diff+'_{0:0>2}'.format(ipar*12)+\
                            '_{0:0>3}-{1:0>3}'.format(int(ps[pindex[0]]),int(ps[pindex[pindex.shape[0]-1]]))+dummy
                        print(('output file: ',oname))
                        poutput = output(output_name = oname, 
                                         output_formats = plotproperties["outputformat"],
                                         output_title= oname,
                                         output_name_first_page_number = "off")



                        plotlist=[spage,poutput]
                        for ip in range(plotproperties["pindex"].shape[0]):
                            plotlist=plotlist+dailyseries(d,data['current'],istnames[istat],lats[istat],lons[istat],istat,ens,ipar,ps,ip,plotproperties,longname=stlongnames[istat])
    #                        
                        try:
                            if os.path.isdir(path+'/'+goodsts[istat]):
                                try:
                                    plot(plotlist)
                                    print((poutput.args['output_name']+'.'+str(plotproperties["outputformat"][0]),time.time()-t))
                                    print(('size:',os.path.getsize(poutput.args['output_name']+'.'+str(plotproperties["outputformat"][0]))))
                                except Exception as e:
                                    print(e)
                                    print(('no time series available:', stnames))
                            else :
                                print(('no dir',poutput.args['output_name']+'.'+str(plotproperties["outputformat"][0]),time.time()-t))
                                
                        except KeyError:
                            print(('could not plot '+poutput.args['output_name']+'.'+str(plotproperties["outputformat"][0]),time.time()-t))
                        try:        
                            del plotlist
                        except:
                            pass

# Now create a "supersonde", i.e. the average of all sondes
        else:
            data['currentav']=numpy.zeros(data['current'].shape[1:4])
            avcount=numpy.zeros(data['current'].shape[1:4])
            d['minindex'][:]=0
            d['maxindex'][:]=data['current'].shape[3]-1
            stationaverage(data['current'],data['currentav'],avcount,d['minindex'],d['maxindex'],3)
            print(('saverage,',time.time()-tt))
            data['current'][0,:,:,:]=data['currentav']
            try:
                os.mkdir(plotproperties['tmppath']+'/'+plotproperties['superstring'])
            except:
                pass
            spage=page(super_page_y_length=pindex.shape[0]*plotproperties['dailyseriesheight']+0.5,
                           super_page_x_length=21.0)

            for ipar in range(2):
                if '{:0>2}'.format(ipar*12) in plotproperties['time']:
#                    oname=plotproperties['tmppath']+'/'+goodsts[0]+'/'+goodsts[0]+'-'+max(goodsts)+'_'+d["shortname"]+diff+'_{0:0>2}'.format(ipar*12)+\
#                        '_{0:0>3}-{1:0>3}'.format(int(ps[pindex[0]]),int(ps[pindex[pindex.shape[0]-1]]))+dummy
                    oname=plotproperties['tmppath']+'/'+plotproperties['superstring']+'/'+plotproperties['superstring']+'_'+d["shortname"]+diff+'_{0:0>2}'.format(ipar*12)+\
                        '_{0:0>3}-{1:0>3}'.format(int(ps[pindex[0]]),int(ps[pindex[pindex.shape[0]-1]]))+dummy
                    print(('output file:',oname))
                    poutput = output(output_name = oname, 
                                     output_formats = plotproperties["outputformat"],
                                     output_title= oname,
                                     output_name_first_page_number = "off")



                    plotlist=[spage,poutput]
        #	    def dailyseries(task,currentdata,istname,lat,lon,istat,ens,ipar,ps,ip,plotproperties):
                    tt=time.time()
                    lgoodsts=numpy.sort(goodsts)
                    for ip in range(plotproperties["pindex"].shape[0]):
                        plotlist=plotlist+dailyseries(d,data['current'],lgoodsts[0]+'-'+lgoodsts[len(goodsts)-1],lats[0],lons[0],0,ens,ipar,ps,ip,plotproperties)
        #           

                    try:
                        if os.path.isdir(path+'/'+plotproperties['superstring']):
                            try:
                                plot(plotlist)
                                print((poutput.args['output_name']+'.'+str(plotproperties["outputformat"][0]),time.time()-t))
                            except:
                                print(('no time series available:', stnames))
                        else :
                            print(('no dir',poutput.args['output_name']+'.'+str(plotproperties["outputformat"][0]),time.time()-t))
                            
                    except KeyError:
                        print(('could not plot '+poutput.args['output_name']+'.'+str(plotproperties["outputformat"][0]),time.time()-t))
                    try:        
                        del plotlist
                    except:
                        pass

    print((time.time()-tt))

    return

    def vertrauensintervall(x,yerr,color='b',alpha=1.,label=None): # yerr ist eine Liste mit den Obergrenzen und Untergrenzen
        poly=[]
        for a in zip(x,yerr[0]):  # zip macht aus den x,yerr Elementen Tupel
            poly.append(a)        # Ein Polygonzug wird erstellt
        for a in reversed(list(zip(x,yerr[1]))):
            poly.append(a)        # Polygonzug mit Obergrenzen
        p=Polygon(poly,facecolor=color,edgecolor='none',alpha=alpha,label=label)
        return p

    xtime=numpy.floor(d['startdate']/10000)+numpy.arange(data['current'].shape[3])/365.25
    ipr=numpy.asarray([[-8,8],[-8,8],[-8,8],[-4,4],[-4,4],[-4,4],[-4,4],[-4,4],[-4,4],[-4,4],[-4,4],[-4,4],[-4,4],[-4,4]])
    for ip in range(data['current'].shape[2]):
        plt.figure(figsize=(8,3))
        ax=plt.gca()
        pdic=OrderedDict()
        for dset,col in zip(('n20c','e20c','ce20c0','bgdep'),('y','r','m','b')):
            if dset in list(data.keys()):
                pdic[dset]=[-data[dset][0,0,ip,:],col,1.0]
        simple=False
        if simple:
            for ids in list(pdic.keys()):
                index=thin2(rmeanw(pdic[ids],30),10)
                plt.plot(xtime[index],rmeanw(pdic[ids][0],30)[index],label=ids+', {:5.2f}'.format(numpy.nanstd(pdic[ids][0])))
        else:
            for ids in list(pdic.keys()):
                il=0
                perc=list()
                xt=[]
                for i in range(0,xtime.shape[0],730):
                    if xtime[i]>plotproperties['plotinterval'][1] or xtime[min(i+730,xtime.shape[0]-1)]<plotproperties['plotinterval'][0]:
                        continue
                    chunk=pdic[ids][0][i:i+730]
                    mask=~numpy.isnan(chunk)
                    if sum(mask)>20:
                        xt.append(xtime[i])
                        perc.append(numpy.percentile(chunk[mask],[25,50,75]))
                    il+=1
                if len(perc)>0:
                    perc=numpy.transpose(numpy.asarray(perc))
                    idxp=(datetime.date(plotproperties['plotinterval'][1],12,31)-datetime.date(1900,1,1)).days
                    idxm=(datetime.date(plotproperties['plotinterval'][0],1,1)-datetime.date(1900,1,1)).days
                    p=vertrauensintervall(xt,perc[[0,2],:],color=pdic[ids][1],alpha=pdic[ids][2],
                                          label=ids+',{0:4.1f},{1:4.1f}'.format(numpy.nanmean(pdic[ids][0][idxm:idxp]),numpy.nanstd(pdic[ids][0][idxm:idxp])))
                    ax.add_patch(p)


        plt.xlim(plotproperties['plotinterval'])
        plt.ylim(ipr[ip,:])
        box = ax.get_position()
        ax.set_position([box.x0*0.8, box.y0, box.width * 0.8, box.height])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=12)
        if len(stnames)<=istat:
            istat=0
        plt.title(stnames[istat]+', '+str(int(ps[pindex[ip]]))+'hPa')
        plt.savefig('Vergleich_'+stnames[istat]+'_'+str(int(ps[pindex[ip]]))+'.pdf')

        plt.close()
    return
