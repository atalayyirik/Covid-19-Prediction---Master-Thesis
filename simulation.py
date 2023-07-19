
#!/usr/bin/python


import json
import os
import subprocess





def simulation(simulationNumber,days):
	

	data = loadOriginalScenario()

	simTimeStep = data['scenario']['attributesSimulation']['simTimeStepLength']
	totalAgentNumber = data['scenario']['topography']['sources'][0]['spawnNumber']
	# For 1st day setting
	numberOfInfectedAtStart = data['scenario']['attributesModel']['org.vadere.state.attributes.models.AttributesSIRG']['numberOfInfected']
	numberOfInfectedAtStart = 2
	# For previous days setting
	simuationCurrentStartInfected = numberOfInfectedAtStart

	scenarioPath = '.\\SimulationFiles\\simulation.scenario' # newly created scenario file path
	outputFilePath = '.\\SimulationOutputs' # output files directory
	nn_file =  open('.//NN_Files//'+data['name']+'_nn_data_'+str(simulationNumber)+'-sim-'+str(days)+'-day'+'.txt','w')
	NN_Output = []
	
	

	for seedNumber in range(1,simulationNumber+1):

		for day in range(1,days+1):

			#print("Day ",day," of Simulation ",seedNumber," Started. ( Total Simulation:",simulationNumber,"-",days," days) ")
			# File names and paths
			fileName = 'SIRInformation_'+str(seedNumber)+'_'+str(day)+'.txt' # output file for SIRInformation_seednumber_day.txt

			# Change fixed Seed,simulation seed,output file name
			data['scenario']['attributesSimulation']['simulationSeed'] = seedNumber
			data['scenario']['attributesSimulation']['fixedSeed'] = seedNumber
			data['processWriters']['files'][0]['filename'] = fileName

			# Change number of SI of previous days
			if(day != 1):
				simuationCurrentStartInfected =NN_Output[-1][1][0]
			else:
				simuationCurrentStartInfected = numberOfInfectedAtStart

			data['scenario']['attributesModel']['org.vadere.state.attributes.models.AttributesSIRG']['numberOfInfected'] = simuationCurrentStartInfected
			
			

			writeNewScenarioFile(data,scenarioPath)
			runNewScenario(scenarioPath,outputFilePath)
			# Get the number of SI for one day
			processed_data,dayData = readSIRData(outputFilePath + '\\'+fileName,simTimeStep,totalAgentNumber,simuationCurrentStartInfected,day,seedNumber,days,simulationNumber)

			
			NN_Output.append(dayData)
			nn_file.write(str(round(dayData[0][0]/totalAgentNumber,2))+' '+str(round(dayData[0][1]/totalAgentNumber,2))+'-'+str(round(dayData[1][0]/totalAgentNumber,2))+' '+str(round(dayData[1][1]/totalAgentNumber,2)) + '\n')
			# no need for writing SIR data
			#writeSIRData(processed_data,outputFilePath + '\\'+'proccessed_'+fileName)
			#print("Day ",day," of Simulation ",seedNumber," Ended. ( Total Simulation:",simulationNumber,"-",days," days) ")
	nn_file.close()



def writeSIRData(processedDataList,output_path):
    SIRFile = open(output_path, 'w')
    for dataPoint in processedDataList:
        SIRFile.write(str(dataPoint[0])+' '+str(dataPoint[1])+' '+str(dataPoint[2])+'\n')
    SIRFile.close()



def loadOriginalScenario():
	# Read JSON File
	with open('../Scenarios/scenarios/Bottleneck.scenario', 'r') as f:
		x =  json.load(f)
	f.close()
	return x


def writeNewScenarioFile(data,scenarioPath):
	# Write a new scenario file to run
	with open(scenarioPath, 'w') as x:
		json.dump(data, x,indent=2)
	x.close()

def runNewScenario(scenarioPath,outputFilePath):
	# Run file from console & receive data points for one day
	
	console = '..\\Vadere-SIR\\vadere-console.jar'
	#command = 'java -jar ..\\Vadere-SIR\\vadere-console.jar suq --output-dir '+outputFilePath+' --scenario-file '+scenarioPath
	#output = os.popen(command).read()
	#subprocess.check_call(['java', '-jar', console,'suq','--output-dir',outputFilePath,'--scenario-file',scenarioPath], stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
	subprocess.check_call(['java', '-jar', console,'suq','--output-dir',outputFilePath,'--scenario-file',scenarioPath], stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
	
	#print(output)


def readSIRData(output_path,simTimeStep=0.4,totalAgentNumber=200,numberOfInfectedAtStart=1,day=1,seedNumber=1,totalDay=1,totalSimulation=1):
    
    SIRFile = open(output_path, 'r')
    SIRData =  SIRFile.readlines()
    columns = SIRData[0].strip('\n').split(' ')



    currentTime = simTimeStep
    current_theta_0 = 0 # number of infected
    current_theta_1 = 0 # number of susceptible
    agent_states = [0] * totalAgentNumber

    processedDataList = []
    dayData = []
    dayData.append([numberOfInfectedAtStart,totalAgentNumber-numberOfInfectedAtStart])
    
    
    

    # Begining of simulation
    for dataPoint in SIRData[1:totalAgentNumber+1]:
        # pedestrianId simTime groupId-PID5
        dataPoint = dataPoint.strip('\n').split(' ')
        if dataPoint[2] == "1":
            agent_states[int(dataPoint[0])-1] = 1
            current_theta_1 +=1
        else:
            agent_states[int(dataPoint[0])-1] = 0
            current_theta_0 +=1

    for dataPoint in SIRData[totalAgentNumber+1:]:
        # pedestrianId simTime groupId-PID5
        dataPoint = dataPoint.strip('\n').split(' ')
        # print(int(dataPoint[0]),float(dataPoint[1]),int(dataPoint[2]))
        
        if dataPoint[2] == "1" and agent_states[int(dataPoint[0])-1]==0:
            agent_states[int(dataPoint[0])-1]=1
            current_theta_1 +=1
            current_theta_0 -=1
        if dataPoint[2] == "0" and agent_states[int(dataPoint[0])-1]==1:
            agent_states[int(dataPoint[0])-1]=0
            current_theta_1 -=1
            current_theta_0 +=1
        if (float (dataPoint[1]))>=currentTime+simTimeStep :
            processedDataList.append([currentTime,current_theta_0,current_theta_1])
            currentTime = round(currentTime + simTimeStep,1)
    dayData.append([current_theta_0,current_theta_1])
    currentTotalSimul = (seedNumber-1)*totalDay +day
    print(f'{day}/{totalDay} of {currentTotalSimul}/{totalSimulation*totalDay} : {numberOfInfectedAtStart} - {totalAgentNumber-numberOfInfectedAtStart} --- {current_theta_0} - {current_theta_1} Finished.')
    #print("Day end :",current_theta_0,current_theta_1)
    # print([currentTime,round(current_theta_0/totalAgentNumber,2),round(1-round(current_theta_0/totalAgentNumber,2),2)])
    # print("--")

    return processedDataList,dayData




def main():



    # all_subdirs = sorted(os.listdir('../Scenarios/output'))
    # folder_name  = all_subdirs[-1] # last updated folder
    # Example folder_path : 'Bottleneck_2022-10-29_14-26-47.659'


    #processedDataList = readSIRData(folder_name,simTimeStep,totalAgentNumber)
    #writeSIRData(processedDataList,folder_name)
    days = 4
    simulationNumber = 1500
    simulation(simulationNumber,days)
    


if __name__ == "__main__":

    main()


# merge data points
# clean and process data
# Have x_data,y_data pairs for NN model