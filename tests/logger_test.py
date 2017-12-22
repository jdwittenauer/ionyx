from ionyx.contrib import Logger

print('Beginning logger test...')

path = '/home/john/temp/log.txt'
logger = Logger(path, mode='replace')
logger.write("Testing replace mode")
logger.close()

logger = Logger(path, mode='append')
logger.write("Testing append mode")
logger.close()

print('Done.')
