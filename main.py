from utils import read_video, save_video
from trackers import Tracker
from shots_on_target import ShotCounter
import numpy as np
from team_assign import TeamAssigner
from player_ball_assign import PlayerBallAssigner
from camera_estimator import CameraMovementEstimator
from view_transformers import ViewTransformer
from speed_and_distance_estimate import SpeedAndDistance_Estimator

def main():
    # Step 1: Read video frames
    video_frames = read_video('input_videos/test (2).mp4')
    
    # Step 2: Initialize Tracker and get object tracks
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    tracker.add_position_to_tracks(tracks)

    # Step 3: Estimate camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Step 4: Transform view to fit perspective
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Step 5: Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Step 6: Estimate player speeds and distances
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Step 7: Assign teams to players
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Step 8: Assign ball possession to players
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
    
    team_ball_control = np.array(team_ball_control)

    # Step 9: Annotate shots on target
    counter = ShotCounter('models/best.pt')
    updated_frames = []
    for frame in video_frames:
        # Call the annotate_shots_on_target function for each frame
        updated_frame = counter.process_video(frame, tracks)
        updated_frames.append(updated_frame)

    # Continue with further processing (drawing tracks, camera movement, saving video, etc.)
    
    # Step 10: Draw object tracks, camera movement, and speed/distance on frames
    output_video_frames = tracker.draw_annotations(updated_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Step 11: Save output video
    save_video(output_video_frames, 'output_videos/output_2.avi')

if __name__ == '__main__':
    main()