#!/usr/bin/python

import Glacier2, Ice, sys

properties = Ice.createProperties(sys.argv)
properties.load('client.cfg')

init_data = Ice.InitializationData()
init_data.properties = properties

Ice.loadSlice('-I%s %s' % (Ice.getSliceDir(), 'MotionDatabase.ice'))
import MotionDatabase

ic = Ice.initialize(init_data)
router = Glacier2.RouterPrx.checkedCast(ic.getDefaultRouter())
session = router.createSession('', '')  # Username and password (leave empty for anonymous access)
db = MotionDatabase.MotionDatabaseSessionPrx.checkedCast(session)

print('db = %s' % db)

print ('db.pingServer("test") = "%s"' % db.pingServer('test'))

print ('db.listInstitutions() = %s' % db.listInstitutions())
# print ('db.getMotionDescriptionTree() = %s' % db.getMotionDescriptionTree())
#
# print ('db.countMotions(None, None, None, None, None, None) = %s' % db.countMotions(None, None, None, None, None, None))
# print('db.listMotions(None, None, None, None, None, None, "id", 3, 0) = %s' % db.listMotions(None, None, None, None, None, None, 'id', 3, 0))
# print ('db.listProjects() = %s' % db.listProjects())
# print ('db.listSubjects() = %s' % db.listSubjects())
# print ('db.listObjects() = %s' % db.listObjects())
#
# print ('db.listFiles(395) = %s' % db.listFiles(395))
#
# # File download is only possible if the user is logged in (provide credentials to router.createSession() in line 16)
# fr = db.getFileReader(6785)
# print ('fr = db.getFileReader(6785) = %s' % fr)
#
# print ('fr.getSize() = %s' % fr.getSize())
# print ('len(fr.readChunk(10)) = %s' % len(fr.readChunk(10)))
# fr.seek(1)
# print('fr.seek(1)')
# print ('len(fr.readChunk(10)) = %s' % len(fr.readChunk(10)))
