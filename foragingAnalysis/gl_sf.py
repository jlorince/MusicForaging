gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 32)
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_GRAPH_LAMBDA_WORKERS', 32)
gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY',100000000000)
gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE',10000000000)
sf = gl.SFrame.read_csv('lastfm_scrobbles.txt',header=None,delimiter='\t',usecols=['X1','X3','X4'])
print 'sframe loaded'
sf.rename({'X1':'user','X3':'artist','X4':'ts'})
print 'sframe renamed'
sf['ts'] = sf['ts'].str_to_datetime()
print 'time stamps converted'
ts = gl.TimeSeries(sf,index='ts')
print 'ts generated'
ts.save('lastfm_scrobbles_gl')
print 'ts saved'
