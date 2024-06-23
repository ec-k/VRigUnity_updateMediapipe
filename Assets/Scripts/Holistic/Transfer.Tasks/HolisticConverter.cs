using Mediapipe.Tasks.Vision.PoseLandmarker;
using System.Collections.Generic;
using UnityEngine;

namespace HardCoded.VRigUnity.Updated
{
	public class HolisticConverter {
		public static void Convert(PoseLandmarkerResult result, IHolisticCallback callback) {
			//// Face
			//graph.OnFaceLandmarksOutput += (_, eventArgs) => {
			//    callback.OnFaceLandmarks(FromMediapipe(eventArgs));
			//};

			//// Hands
			//graph.OnLeftHandLandmarksOutput += (_, eventArgs) => {
			//    callback.OnLeftHandLandmarks(FromMediapipe(eventArgs));
			//};
			//graph.OnRightHandLandmarksOutput += (_, eventArgs) => {
			//    callback.OnRightHandLandmarks(FromMediapipe(eventArgs));
			//};

			// Pose
			callback.OnPoseLandmarks(FromMediapipe(result, false));
			callback.OnPoseWorldLandmarks(FromMediapipe(result));
		}

		public static HolisticLandmarks FromMediapipe(PoseLandmarkerResult result, bool modify = true) {
			var landmarkList = result.poseLandmarks;
			if (landmarkList == null) {
				return HolisticLandmarks.NotPresent;
			}
			
			List<Vector4> list = new();
			List<Vector4> raw = new();
			int count = landmarkList[0].landmarks.Count;
			for (int i = 0; i < count; i++) {
				var mark = landmarkList[0].landmarks[i];
				if (modify) {
					list.Add(new(mark.x * 2, mark.y, mark.z * 2, mark.visibility??0));
				} else {
					list.Add(new(mark.x, mark.y, mark.z, mark.visibility??0));
				}
				raw.Add(new(mark.x, mark.y, mark.z, mark.visibility??0));
			}

			return new(list, raw);
		}

		public static HolisticLandmarks FromMediapipe(PoseLandmarkerResult result) {
			var landmarkList = result.poseWorldLandmarks;
			if (landmarkList == null) {
				return HolisticLandmarks.NotPresent;
			}
			
			List<Vector4> list = new();
			int count = landmarkList[0].landmarks.Count;
			for (int i = 0; i < count; i++) {
				var mark = landmarkList[0].landmarks[i];
				list.Add(new(mark.x, mark.y, mark.z, mark.visibility??0));
			}

			return new(list, list);
		}
	}
}
