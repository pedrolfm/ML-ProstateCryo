import os
import unittest
import shutil
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy
import scipy.misc
import imageio
#
# Cryo
#
NOFSLICES = 17
X = 192
Y = 168
CASE = 'Case_16'

class Cryo(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Cryo" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["John Doe (AnyWare Corp.)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# CryoWidget
#

class CryoWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    slicer.util.mainWindow().moduleSelector().selectModule('Markups')

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input volume selector
    #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Input Volume: ", self.inputSelector)

    #
    # output volume selector
    #
    self.outputSelector = slicer.qMRMLNodeComboBox()
    self.outputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.outputSelector.selectNodeUponCreation = True
    self.outputSelector.addEnabled = True
    self.outputSelector.removeEnabled = True
    self.outputSelector.noneEnabled = True
    self.outputSelector.showHidden = False
    self.outputSelector.showChildNodeTypes = False
    self.outputSelector.setMRMLScene( slicer.mrmlScene )
    self.outputSelector.setToolTip( "Pick the output to the algorithm." )
    parametersFormLayout.addRow("Output Volume: ", self.outputSelector)

    #
    # threshold value
    #
    self.imageThresholdSliderWidget = ctk.ctkSliderWidget()
    self.imageThresholdSliderWidget.singleStep = 0.1
    self.imageThresholdSliderWidget.minimum = -100
    self.imageThresholdSliderWidget.maximum = 100
    self.imageThresholdSliderWidget.value = 0.5
    self.imageThresholdSliderWidget.setToolTip("Set threshold value for computing the output image. Voxels that have intensities lower than this value will set to zero.")
    parametersFormLayout.addRow("Image threshold", self.imageThresholdSliderWidget)

    #
    # check box to trigger taking screen shots for later use in tutorials
    #
    self.enableScreenshotsFlagCheckBox = qt.QCheckBox()
    self.enableScreenshotsFlagCheckBox.checked = 0
    self.enableScreenshotsFlagCheckBox.setToolTip("If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
    parametersFormLayout.addRow("Enable Screenshots", self.enableScreenshotsFlagCheckBox)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.inputSelector.currentNode() and self.outputSelector.currentNode()

  def onApplyButton(self):
    logic = CryoLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    imageThreshold = self.imageThresholdSliderWidget.value
    logic.run(self.inputSelector.currentNode(), self.outputSelector.currentNode(), imageThreshold, enableScreenshotsFlag)

#
# CryoLogic
#

class CryoLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def hasImageData(self,volumeNode):
    """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() is None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def isValidInputOutputData(self, inputVolumeNode, outputVolumeNode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not outputVolumeNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False
    if inputVolumeNode.GetID()==outputVolumeNode.GetID():
      logging.debug('isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
      return False
    return True

  def takeScreenshot(self,name,description,type=-1):
    # show the message even if not taking a screen shot
    slicer.util.delayDisplay('Take screenshot: '+description+'.\nResult is available in the Annotations module.', 3000)

    lm = slicer.app.layoutManager()
    # switch on the type to get the requested window
    widget = 0
    if type == slicer.qMRMLScreenShotDialog.FullLayout:
      # full layout
      widget = lm.viewport()
    elif type == slicer.qMRMLScreenShotDialog.ThreeD:
      # just the 3D window
      widget = lm.threeDWidget(0).threeDView()
    elif type == slicer.qMRMLScreenShotDialog.Red:
      # red slice window
      widget = lm.sliceWidget("Red")
    elif type == slicer.qMRMLScreenShotDialog.Yellow:
      # yellow slice window
      widget = lm.sliceWidget("Yellow")
    elif type == slicer.qMRMLScreenShotDialog.Green:
      # green slice window
      widget = lm.sliceWidget("Green")
    else:
      # default to using the full window
      widget = slicer.util.mainWindow()
      # reset the type so that the node is set correctly
      type = slicer.qMRMLScreenShotDialog.FullLayout

    # grab and convert to vtk image data
    qimage = ctk.ctkWidgetsUtils.grabWidget(widget)
    imageData = vtk.vtkImageData()
    slicer.qMRMLUtils().qImageToVtkImageData(qimage,imageData)

    annotationLogic = slicer.modules.annotations.logic()
    annotationLogic.CreateSnapShot(name, description, type, 1, imageData)

  def GetDistances3(self,Pr1,Pr2,Pr3,p,Uretra):
    #TODO: Clean the code.... reduce number of variables - self?
    
    #par = [4.1, -0.006, -0.0038, 0.08, -1.1, 0.4, 0.4, 0.06]
    par = [3.907298969, -0.005854036, -0.003720409, 0.059606987]
  

    distanceProbes = numpy.sqrt((Pr1[0] - Pr2[0])*(Pr1[0] - Pr2[0])+(Pr1[1] - Pr2[1])*(Pr1[1] - Pr2[1])+(Pr1[2] - Pr2[2])*(Pr1[2] - Pr2[2]))

    offset = 3.0
    distance1 = [0, 0, 0]
    distance2 = [0, 0, 0]
    distance3 = [0, 0, 0]
    distance4 = [0, 0, 0]
    distance1[0] = (-Pr1[0] + p[0])*(-Pr1[0] + p[0])
    distance1[1] = (-Pr1[1] + p[1])*(-Pr1[1] + p[1])
    distance1[2] = (-Pr1[2] + p[2])#*(Pr1[2] - p[2])
    distance1a = (Pr1[2]- offset - p[2]) * (Pr1[2]-offset - p[2])
    distance1b = (Pr1[2] + offset - p[2]) * (Pr1[2] + offset - p[2])

    distance2[0] = (Pr2[0] - p[0])*(Pr2[0] - p[0])
    distance2[1] = (Pr2[1] - p[1])*(Pr2[1] - p[1])
    distance2[2] = (-Pr2[2] + p[2])#*(Pr2[2] - p[2])
    distance2a = (Pr2[2] - offset - p[2]) * (Pr2[2] - offset - p[2])
    distance2b = (Pr2[2] + offset - p[2]) * (Pr2[2] + offset - p[2])
    distance3[0] = (Uretra[0] - p[0])*(Uretra[0] - p[0])
    distance3[1] = (Uretra[1] - p[1])*(Uretra[1] - p[1])
    distance3[2] = (Uretra[2] - p[2])*(Uretra[2] - p[2])
    
    distance4[0] = (Pr3[0] - p[0])*(Pr3[0] - p[0])
    distance4[1] = (Pr3[1] - p[1])*(Pr3[1] - p[1])
    distance4[2] = (-Pr3[2] + p[2])#*(Pr2[2] - p[2])

    suma = numpy.sqrt(distance1[0] + distance1[1])+numpy.sqrt(distance2[0] + distance2[1])+numpy.sqrt(distance4[0] + distance4[1])
    suma = suma*suma/1.0
    
    suma2 = numpy.abs(distance1[2])+numpy.abs(distance2[2])+numpy.abs(distance4[2])
    suma2 = suma2*suma2/1.0
#MODEL 1
    value = par[0] + par[1]*suma \
    + par[2] * suma2 \
            + par[3] * numpy.sqrt(distance3[0] + distance3[1] + distance3[2])
    if numpy.sqrt(distance3[0] + distance3[1] + distance3[2]) > 5:
      return numpy.exp(value)/(1+numpy.exp(value))
    else:
      return 0
  
  def GetDistances(self,Pr1,Pr2,p,Uretra):
    #TODO: Clean the code.... reduce number of variables - self?
    
    #par = [3.950076, -0.0059, -0.003810, 0.061219] #CASE10
    #par = [4.03773698, -0.006163,-0.0038245, 0.061118646] # CASE17
    par = [4.091433315, -0.005919230, -0.003808784,  0.052071316]

    distanceProbes = numpy.sqrt((Pr1[0] - Pr2[0])*(Pr1[0] - Pr2[0])+(Pr1[1] - Pr2[1])*(Pr1[1] - Pr2[1])+(Pr1[2] - Pr2[2])*(Pr1[2] - Pr2[2]))

    distance1 = [0, 0, 0]
    distance2 = [0, 0, 0]
    distance3 = [0, 0, 0]
    distance4 = [0, 0, 0]
    distance1[0] = (-Pr1[0] + p[0])*(-Pr1[0] + p[0])
    distance1[1] = (-Pr1[1] + p[1])*(-Pr1[1] + p[1])
    distance1[2] = (-Pr1[2] + p[2])#*(Pr1[2] - p[2])

    distance2[0] = (Pr2[0] - p[0])*(Pr2[0] - p[0])
    distance2[1] = (Pr2[1] - p[1])*(Pr2[1] - p[1])
    distance2[2] = (-Pr2[2] + p[2])#*(Pr2[2] - p[2])

    distance3[0] = (Uretra[0] - p[0])*(Uretra[0] - p[0])
    distance3[1] = (Uretra[1] - p[1])*(Uretra[1] - p[1])
    distance3[2] = (Uretra[2] - p[2])*(Uretra[2] - p[2])
  
    suma = numpy.sqrt(distance1[0] + distance1[1])+numpy.sqrt(distance2[0] + distance2[1])
    suma = suma*suma/1.0
    
    suma2 = numpy.abs(distance1[2])+numpy.abs(distance2[2])
    suma2 = suma2*suma2/1.0
#MODEL 1
    value = par[0] + par[1]*suma \
    + par[2] * suma2 \
            + par[3] * numpy.sqrt(distance3[0] + distance3[1] + distance3[2])
    if numpy.sqrt(distance3[0] + distance3[1] + distance3[2]) > 5:
      return numpy.exp(value)/(1+numpy.exp(value))
    else:
      return 0

  def getLimits(self,probes,rasToijkMatrix):
    numProbes = probes.size/3
    limitX = numpy.arange(numProbes)
    limitY = numpy.arange(numProbes)
    limitZ = numpy.arange(numProbes)

    for i in range(0, numProbes):
      pos_temp = [0,0,0,1]
      pos_temp = rasToijkMatrix.MultiplyDoublePoint([probes[i,0], probes[i,1], probes[i,2], 1])
      limitX[i] = pos_temp[0]
      limitY[i] = pos_temp[1]
      limitZ[i] = pos_temp[2]

    lim = numpy.array([numpy.amin(limitX)-30, numpy.amax(limitX)+30,numpy.amin(limitY)-30, numpy.amax(limitY)+30,numpy.amin(limitZ)-5, numpy.amax(limitZ)+5])
    if lim[1] >= 190:
      lim[1] = 190
    if lim[3] >= 168:
      lim[3] = 168
    if lim[5] >= 19:
      lim[5] = 19
    if lim[4] < 1:
      lim[4] = 1

    #TEST FOR HEAT MAP
    lim[0]=0
    lim[1]=X-1
    lim[2]=0
    lim[3]=Y-1
    lim[4]=1
    lim[5]=21
    print("limits:")
    print(lim)
    return lim

  def run(self, inputVolume, outputVolume, imageThreshold, enableScreenshots=0):

    #Pr1 = [-19.8, 1.6, 62.7]
    #Pr2 = [-31.5, 10.8, 62.7]
    Pr2 = [-8.5, 4.1, 4.3]
    Pr1 = [-8.6, 11.8, 0.7]
    Uretra = [89, 79, 13, 1]

    UretraMatrix = numpy.matrix([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
    probe1 = numpy.matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    probe2 = numpy.matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    probe3 = numpy.matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


    fidList = slicer.util.getNode('urethra')
    numFids = fidList.GetNumberOfFiducials()
    print(numFids)
    for i in range(numFids):
      ras = [0, 0, 0]
      fidList.GetNthFiducialPosition(i, ras)
      UretraMatrix[i,] = ras

    fidList2 = slicer.util.getNode("probe1")
    numFids2 = fidList2.GetNumberOfFiducials()
    for i in range(numFids2):
      ras = [0, 0, 0]
      fidList2.GetNthFiducialPosition(i, ras)
      probe1[i,] = ras
      probe1[i,2] = probe1[i,2] - 10.0


    fidList3 = slicer.util.getNode('probe2')
    numFids3 = fidList2.GetNumberOfFiducials()
    for i in range(numFids3):
      ras = [0, 0, 0]
      fidList3.GetNthFiducialPosition(i, ras)
      probe2[i,] = ras
      probe2[i,2] = probe2[i,2] - 10.0
    
    nofProbes = 2
    try:
      fidList4 = slicer.util.getNode('probe3')
      numFids4 = fidList2.GetNumberOfFiducials()
      for i in range(numFids4):
        ras = [0, 0, 0]
        fidList4.GetNthFiducialPosition(i, ras)
        probe3[i,] = ras
        probe3[i,2] = probe3[i,2] - 10.0
      nofProbes = 3
    except:
      nofProbes = 2
      
    probes = numpy.concatenate((probe1[0,], probe2[0,]), axis=0)
    print(probes)
    print('====Files loaded====')
    print(nofProbes)



    resultFileName = "Table-FullImage-1y"
    resultFilePath = '/Users/pedro/Projects/MLCryo/Cases' + '/' + resultFileName
    #resultFile = open(resultFilePath, 'a')

    #resultFile.write("value; PosX; PosY ; PosZ; pr1X; Pr1Y ; Pr1Z; pr2X; Pr2Y ; Pr2Z ; Urx ; Ury ; Urz \n")

    IjkToRasMatrix = vtk.vtkMatrix4x4()
    inputVolume.GetIJKToRASMatrix(IjkToRasMatrix)

    #Be careful voxel array changes x for z
    voxelArray = slicer.util.arrayFromVolume(inputVolume)

    nOfSlices = NOFSLICES-1

    imageData = vtk.vtkImageData()
    imageData.SetDimensions(X, Y, nOfSlices)
    print(inputVolume.GetSpacing())
    imageData.SetSpacing(0.9, 0.9, 3.6)
    imageData.SetOrigin(72, 83, 26)
    imageData.AllocateScalars(vtk.VTK_FLOAT, 1)

    img3 = numpy.zeros([nOfSlices, Y, X])
    img4 = numpy.zeros([nOfSlices, Y, X])
    img2 = numpy.zeros([nOfSlices, Y, X])
    probs = numpy.zeros([7000000, 1, 1])

    rasToijkMatrix = vtk.vtkMatrix4x4()
    inputVolume.GetRASToIJKMatrix(rasToijkMatrix)
    lim = self.getLimits(probes, rasToijkMatrix)

    rgb = numpy.zeros((Y,X, 3), dtype=numpy.uint8)
    f=0
    for k in range(0, nOfSlices):
      for i in range(lim[0], lim[1]):
        for j in range(lim[2], lim[3]):
            p = IjkToRasMatrix.MultiplyDoublePoint([i, j, k, 1])
            #resultFile.write("%02d; %02f ; %02f ; %02f ; %02f ; %02f ; %02f ; %02f ; %02f ; %02f; %02f ; %02f ; %02f\n" % (voxelArray[k,j,i],p[0],p[1],p[2],probe1[0,0],probe1[0,1],probe1[0,2],probe2[0,0],probe2[0,1],probe2[0,2],UretraMatrix[k,0],UretraMatrix[k,1],UretraMatrix[k,2]))
            Pr1 = [probe1[0,0],probe1[0,1],probe1[0,2]]
            Pr2 = [probe2[0,0],probe2[0,1],probe2[0,2]]
            Pr3 = [probe3[0,0],probe3[0,1],probe3[0,2]]
            Uretra = [UretraMatrix[k,0],UretraMatrix[k,1],UretraMatrix[k,2]]
            if nofProbes == 3:
              prob = self.GetDistances3(Pr1,Pr2,Pr3,p,Uretra)
            else:
              prob = self.GetDistances(Pr1,Pr2,p,Uretra)
            imageData.SetScalarComponentFromFloat(k, j, i, 0, 1.0*prob)
            img2[k, j, i] = 1.0*prob
            img4[k, j, i] = voxelArray[k,j,i]
            probs[f,0,0] = prob
            f = f+1
            
            temp = (p[0]-probe1[0,0])*(p[0]-probe1[0,0])+(p[1]-probe1[0,1])*(p[1]-probe1[0,1])+(p[2]-probe1[0,2])*(p[2]-probe1[0,2])
            temp2 = (p[0]-probe2[0,0])*(p[0]-probe2[0,0])+(p[1]-probe2[0,1])*(p[1]-probe2[0,1])+(p[2]-probe2[0,2])*(p[2]-probe2[0,2])
            tempZ = (p[2]-probe1[0,2])*(p[2]-probe1[0,2])+(p[2]-probe2[0,2])*(p[2]-probe2[0,2])
            temp3 = (p[0]-Uretra[0])*(p[0]-Uretra[0])+(p[1]-Uretra[1])*(p[1]-Uretra[1])
            rgb[j, i, 2] = 40*(numpy.log(numpy.sqrt(temp)))
            rgb[j, i, 1] = 40*(numpy.log(numpy.sqrt(temp2)))
            
            if (numpy.abs(tempZ) < 1.0):
              tempZ = 1
            rgb[j, i, 0] = 215-20*numpy.log(numpy.abs(tempZ))#255/numpy.sqrt(temp3)
            if (temp3<5):
              rgb[j, i, 0] = 255
              rgb[j, i, 1] = 255
              rgb[j, i, 2] = 255

      print(rgb[100, 100, 0])
      Case = CASE
      filename = '/Users/pedro/Dropbox (Partners HealthCare)/DeepLearningCryo/New' + '/' + Case + '/' + str(k) + '.png'
      imageio.imwrite(filename, rgb)
      img1 = 255*voxelArray[k,0:Y-1,0:X-1] 
      filename2 = '/Users/pedro/Dropbox (Partners HealthCare)/DeepLearningCryo/New' + '/' + Case + '/LabeL_' + str(k) + '.png'
      imageio.imwrite(filename2, img1)          
        
        
    print("limites")
    
    volumeNode = slicer.vtkMRMLLabelMapVolumeNode()
    slicer.util.updateVolumeFromArray(volumeNode, img3)

    volumeNode2 = slicer.vtkMRMLLabelMapVolumeNode()
    slicer.util.updateVolumeFromArray(volumeNode2, img2)
    volumeNode2.SetSpacing(0.3, 0.3, 3.6)
    volumeNode2.SetIJKToRASMatrix(IjkToRasMatrix)
    volumeNode2.SetName("ProbMap")
    slicer.mrmlScene.AddNode(volumeNode2)

    volumeNodeImage = slicer.vtkMRMLScalarVolumeNode()#slicer.vtkMRMLLabelMapVolumeNode()
    slicer.util.updateVolumeFromArray(volumeNodeImage, img2)

    displayNode = slicer.vtkMRMLScalarVolumeDisplayNode()


    volumeNodeImage.SetAndObserveDisplayNodeID(displayNode.GetID())
    volumeNode.SetSpacing(0.3, 0.3, 3.6)
#    volumeNodeImage.SetOrigin(72, 83, 26)
    volumeNodeImage.SetIJKToRASMatrix(IjkToRasMatrix)
    volumeNode.SetIJKToRASMatrix(IjkToRasMatrix)
    slicer.mrmlScene.AddNode(volumeNodeImage)
    slicer.mrmlScene.AddNode(volumeNode)
    slicer.mrmlScene.AddNode(displayNode)
    displayNode.SetAndObserveColorNodeID('vtkMRMLColorTableNodeGrey')

    if not self.isValidInputOutputData(inputVolume, outputVolume):
      slicer.util.errorDisplay('Input volume is the same as output volume. Choose a different output volume.')
      return False

    logging.info('Processing started')

    # Compute the thresholded output volume using the Threshold Scalar Volume CLI module
    cliParams = {'InputVolume': inputVolume.GetID(), 'OutputVolume': outputVolume.GetID(), 'ThresholdValue' : imageThreshold, 'ThresholdType' : 'Above'}
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True)

    # Capture screenshot
    if enableScreenshots:
      self.takeScreenshot('CryoTest-Start','MyScreenshot',-1)

    logging.info('Processing completed')

    return True


class CryoTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_Cryo1()

  def test_Cryo1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import urllib
    downloads = (
        ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
        )

    for url,name,loader in downloads:
      filePath = slicer.app.temporaryPath + '/' + name
      if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
        logging.info('Requesting download %s from %s...\n' % (name, url))
        urllib.urlretrieve(url, filePath)
      if loader:
        logging.info('Loading %s...' % (name,))
        loader(filePath)
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = CryoLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
