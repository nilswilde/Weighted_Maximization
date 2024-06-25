from aiohttp import web
import socketio
from experiments import Experiment
import os.path
import random
import numpy as np

experiments = {}

def active_experiment_id(experiments):
    res = 0
    for experiment in experiments:
        if experiment.done:
            res += 1
            continue
        return res
    return None # this means all experiments have already been completed

def question_id(experiments):
    return int(np.sum([experiment.overall_id for experiment in experiments]) + 1)

sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

async def form(request):
    path = request._message.path[2:]
    if len(path) < 13 or path[4:13].lower() != 'username=':
        filename = 'invalidurl.html'
    else:
        username = path[4:13]
        if len(username) == 0:
            filename = 'invalidusername.html'
        else:
            filename = 'form.html'
    with open(filename) as f:
        return web.Response(text=f.read(), content_type='text/html')

async def index(request):
    path = request._message.path[2:]
    if len(path) < 9 or path[:9].lower() != 'username=':
        filename = 'invalidurl.html'
    else:
        username = path[9:]
        if len(username) == 0:
            filename = 'invalidusername.html'
        else:
            if username not in experiments.keys(): # new user
                experiments[username] = [Experiment('random',0.1), Experiment('information',0.1), Experiment('random',1.0), Experiment('random', 0.1)]
                random.shuffle(experiments[username])
                filename = 'index.html'
            elif active_experiment_id(experiments[username]) is None: # this user has already completed all the experiments
                filename = 'alreadycompleted.html'
            else: # this user has started, but not finished
                filename = 'index.html'
    with open(filename) as f:
        return web.Response(text=f.read(), content_type='text/html')

async def video(request):
    path = request._message.path[7:]
    if len(path) < 5 or path[:5].lower() != 'name=':
        filename = 'invalidurl.html'
    else:
        video_name = path[5:]
        if len(video_name) == 0:
            assert False, 'Invalid video! How did this happen?'
        else:
            if os.path.exists('videos/' + video_name + '.mp4'):
                filename = 'videos/' + video_name + '.mp4'
                with open(filename, 'rb') as f:
                    return web.Response(body=f.read(), content_type='video/mp4')
            else:
                filename = 'videos/' + video_name + '.webm'
                with open(filename, 'rb') as f:
                    return web.Response(body=f.read(), content_type='video/webm')
        
async def gif(request):
    path = request._message.path[5:]
    if len(path) < 5 or path[:5].lower() != 'name=':
        filename = 'invalidurl.html'
    else:
        gif_name = path[5:]
        if len(gif_name) == 0:
            assert False, 'Invalid gif! How did this happen?'
        else:
            filename = 'web_imgs/' + gif_name + '.gif'
    with open(filename, 'rb') as f:
        return web.Response(body=f.read(), content_type='image/gif')

@sio.on('sendAction')
async def receive_action(sid, message):
    print(message['username'] + ' has sent the following data: ' + message['data'])
    e_id = active_experiment_id(experiments[message['username']])
    experiments[message['username']][e_id].receive_feedback(float(message['data']))
    print('This is the #' + str(experiments[message['username']][e_id].overall_id) + ' data I am receiving from ' + message['username'])
    if not experiments[message['username']][e_id].done:
        print('I am creating a new query for ' + message['username'])
        id_input1, id_input2, resolution = experiments[message['username']][e_id].optimize_query()
        if id_input2 is not None:
            await sio.emit('newQuery', {'option0': experiments[message['username']][e_id].task + str(id_input1), 'option1': experiments[message['username']][e_id].task + str(id_input2), 'resolution': resolution, 'question_id': question_id(experiments[message['username']])}, to=sid)
        else:
            await sio.emit('ordinalQuery', {'option': experiments[message['username']][e_id].task + str(id_input1), 'question_id': question_id(experiments[message['username']])}, to=sid)
    else:
        if e_id + 1 < len(experiments[message['username']]):
            e_id += 1
            print(message['username'] + ' is starting the phase ' + str(e_id) + ' of the experiment.')
            print('I am creating a new query for ' + message['username'])
            id_input1, id_input2, resolution = experiments[message['username']][e_id].optimize_query()
            if id_input2 is not None:
                await sio.emit('newQuery', {'option0': experiments[message['username']][e_id].task + str(id_input1), 'option1': experiments[message['username']][e_id].task + str(id_input2), 'resolution': resolution, 'question_id': question_id(experiments[message['username']])}, to=sid)
            else:
                await sio.emit('ordinalQuery', {'option': experiments[message['username']][e_id].task + str(id_input1), 'question_id': question_id(experiments[message['username']])}, to=sid)
        else:            
            print(message['username'] + ' has completed the experiment.')
            for i in range(len(experiments[message['username']])):
                experiments[message['username']][i].save(message['username'], i)
            print('I saved all the data for ' + message['username'])
            await sio.emit('experimentOver', {'username': message['username']}, to=sid)

@sio.on('start')
async def start(sid, message):
    e_id = active_experiment_id(experiments[message['username']])
    print(message['username'] + ' is starting the experiment. They are in phase ' + str(e_id))
    print('I am creating a new query for ' + message['username'])
    id_input1, id_input2, resolution = experiments[message['username']][e_id].optimize_query()
    if id_input2 is not None:
        await sio.emit('newQuery', {'option0': experiments[message['username']][e_id].task + str(id_input1), 'option1': experiments[message['username']][e_id].task + str(id_input2), 'resolution': resolution, 'question_id': question_id(experiments[message['username']])}, to=sid)
    else:
        await sio.emit('ordinalQuery', {'option': experiments[message['username']][e_id].task + str(id_input1), 'question_id': question_id(experiments[message['username']])}, to=sid)

# We bind our aiohttp endpoint to our app router
app.router.add_get('/form', form)
app.router.add_get('/', index)
app.router.add_get('/video', video)
app.router.add_get('/gif', gif)

# We kick off our server
if __name__ == '__main__':
    web.run_app(app, port=8080)