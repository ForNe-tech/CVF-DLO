using Newtonsoft.Json;
using System;
using System.IO;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class RouteProjectManager : MonoBehaviour
{
    public static RouteProjectManager Instance;

    private static int IMG_Width = 896;
    private static int IMG_Height = 504;
    private static Color32[] pixels;
    private static Matrix4x4 cameraToWorldMatrix;
    private static Matrix4x4 projectionMatrix;

    public GameObject Scene;

    private bool once = true;

    private string totalDir;

    private Color randomColor = Color.red;

    public int layerID = 8;
    public LayerMask LayerCollider;

    // Start is called before the first frame update
    void Start()
    {
        totalDir = @"G:/Research/DLOs Detection/CVF3D-DLO-main/data/LAB_imgs_design_DLO";
        GetScenePose(totalDir + "/ScenePose.json");
    }

    // Update is called once per frame
    void Update()
    {
        if (once)
        {
            once = false;
            Debug.Log("Start Project!");
            string[] filePaths = Directory.GetFiles(totalDir + "/path2D_bs");
            foreach (string filePath in filePaths)
            {
                string pngName = Path.GetFileName(filePath);
                string[] parts = pngName.Split('_');
                string fileName = parts[0];
                string fileType = parts[1];
                if (fileType[0] == 'p')
                {
                    ProjectPathToSpace(totalDir, fileName, Color.red);
                }
            }
            //ProjectRouteToSpace(totalDir, "temp_0.jpg", "temp", Color.red);
            //ProjectRouteToSpace(totalDir, "temp0_0.jpg", "temp0", Color.green);
            //ProjectPathToSpace(totalDir, "temp20", Color.red);
        }
    }

    public void GetScenePose(string jsonPath)
    {
        string jsonContent = File.ReadAllText(jsonPath);
        Dictionary<string, string> Scene_Pose = JsonToDictionary(jsonContent);
        Vector3 Scene_P = StringToVector3(Scene_Pose["Scene_P"].ToString());
        Quaternion Scene_Q = StringToQuaternion(Scene_Pose["Scene_Q"].ToString());
        // Vector3 Scene_Scale = new Vector3(1.008004f, 1.008004f, 1.008004f);
        Scene.transform.position = Scene_P;
        Scene.transform.rotation = Scene_Q;
    }

    public static Vector3 PixelCoordToWorldCoord(Matrix4x4 cameraToWorldMatrix, Matrix4x4 projectionMatrix, Vector2 pixelCoordinates)
    {
        pixelCoordinates = ConvertPixelCoordsToScaleCoords(pixelCoordinates);

        float focalLengthX = projectionMatrix.GetColumn(0).x;
        float focalLengthY = projectionMatrix.GetColumn(1).y;
        float centerX = projectionMatrix.GetColumn(2).x;
        float centerY = projectionMatrix.GetColumn(2).y;

        float normFactor = projectionMatrix.GetColumn(2).z;
        centerX = centerX / normFactor;
        centerY = centerY / normFactor;

        Vector3 dirRay = new Vector3((pixelCoordinates.x - centerX) / focalLengthX, (pixelCoordinates.y - centerY) / focalLengthY, 1.0f / normFactor);
        Vector3 direction = new Vector3(Vector3.Dot(cameraToWorldMatrix.GetRow(0), dirRay), Vector3.Dot(cameraToWorldMatrix.GetRow(1), dirRay), Vector3.Dot(cameraToWorldMatrix.GetRow(2), dirRay));

        return direction;
    }

    public static Vector2 ConvertPixelCoordsToScaleCoords(Vector2 pixelCoords)
    {
        float halfWidth = IMG_Width / 2f;
        float halfHeight = IMG_Height / 2f;

        pixelCoords.x -= halfWidth;
        pixelCoords.y -= halfHeight;

        pixelCoords = new Vector2(pixelCoords.x / halfWidth, pixelCoords.y / halfHeight * -1f);

        return pixelCoords;
    }

    public Vector3 ProjectPointToSpace(Vector3 ori, Vector3 dir, Color color)
    {
        CreateRayEntity(ori, dir, color);
        RaycastHit hit;
        Vector3 hitpoint = new Vector3(0.0f, 0.0f, 0.0f);
        Debug.Log("Project Here");
        if (Physics.Raycast(ori, dir, out hit, Mathf.Infinity, LayerCollider))
        {
            // 如果射线与平面碰撞，打印碰撞物体信息
            Debug.Log("Hit Point:" + hit.point.ToString("F6"));
            // 将碰撞点保存到json文件中
            hitpoint = hit.point;
            // 在场景中绘制空间点
            CreateSphere(hitpoint);
            
        }
        else
        {
            Debug.Log("Don't Hit");
        }

        return hitpoint;
    }

    public void CreateRayEntity(Vector3 ori, Vector3 dir, Color color)
    {
        GameObject RayForShow = new GameObject("RayForShow");
        RayForShow.layer = layerID;

        RayForShow.AddComponent<LineRenderer>();
        LineRenderer lineRenderer = RayForShow.GetComponent<LineRenderer>();
        lineRenderer.startColor = color;
        lineRenderer.endColor = color;
        lineRenderer.startWidth = 0.001f;
        lineRenderer.endWidth = 0.001f;
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        lineRenderer.positionCount = 2;
        lineRenderer.SetPosition(0, ori);
        lineRenderer.SetPosition(1, ori + dir.normalized * 1f);

        //RayForShow.transform.position = ori;
        //RayForShow.transform.rotation = Quaternion.LookRotation(dir);
        //BoxCollider collider = RayForShow.AddComponent<BoxCollider>();
        //collider.size = new Vector3(0.0001f, 0.0001f, 0.5f);
        //collider.center = new Vector3(0f, 0f, 0.5f);
    }

    public static Vector3 GetHololensCameraPosByMatrix(Matrix4x4 cameraToWorldMatrix)
    {
        Vector3 pos = new Vector3(cameraToWorldMatrix[12], cameraToWorldMatrix[13], cameraToWorldMatrix[14]);
        return pos;
    }

    public static void CreateSphere(Vector3 point)
    {
        GameObject SpacePoint = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        SpacePoint.GetComponent<Collider>().enabled = false;
        SpacePoint.transform.localScale = new Vector3(0.01f, 0.01f, 0.01f);
        SpacePoint.transform.position = point;
    }

    private static List<Vector3> RouteImageToDirections(string imgPath, string jsonPath)
    {
        string jsonContent = File.ReadAllText(jsonPath);
        Dictionary<string, string> Pose_Json = JsonToDictionary(jsonContent);
        cameraToWorldMatrix = StringToMatrix(Pose_Json["cameraToWorldMatrix"].ToString());
        projectionMatrix = StringToMatrix(Pose_Json["projectionMatrix"].ToString());

        Texture2D texture2D = new Texture2D(IMG_Width, IMG_Height);
        byte[] fileData = System.IO.File.ReadAllBytes(imgPath);
        texture2D.LoadImage(fileData);

        List<Vector3> directions = new List<Vector3>();

        if (texture2D != null)
        {
            pixels = texture2D.GetPixels32();
            // Debug.Log("Pixels:" + pixels);

            for (int i = 0; i < pixels.Length; i++)
            {
                // Debug.Log("pixels[" + i.ToString() + "]:" + pixels[i]);
                if (pixels[i].r != 0)
                {
                    int pixel_x = i % texture2D.width;
                    int pixel_y = i / texture2D.width;
                    Vector2 pixelCoord = new Vector2(pixel_x, IMG_Height - pixel_y);
                    Debug.Log("pixelCoord:" + pixelCoord.x.ToString() + pixelCoord.y.ToString());
                    Vector3 direction = PixelCoordToWorldCoord(cameraToWorldMatrix, projectionMatrix, pixelCoord);
                    Debug.Log("direction:" + direction.x.ToString() + direction.y.ToString() + direction.z.ToString());
                    directions.Add(direction);
                }
            }
        }

        return directions;
    }

    private static Dictionary<string, List<Vector3>> PathJsonToDirections(string dloPath, string jsonPath)
    {
        string jsonContent = File.ReadAllText(jsonPath);
        Dictionary<string, string> Pose_Json = JsonToDictionary(jsonContent);
        cameraToWorldMatrix = StringToMatrix(Pose_Json["cameraToWorldMatrix"].ToString());
        projectionMatrix = StringToMatrix(Pose_Json["projectionMatrix"].ToString());

        string dloContent = File.ReadAllText(dloPath);
        Dictionary<string, List<List<float>>> Path_Dict = JsonConvert.DeserializeObject<Dictionary<string, List<List<float>>>>(dloContent);

        Dictionary<string, List<Vector3>> Dirs_Dict = new Dictionary<string, List<Vector3>>();
        foreach (KeyValuePair<string, List<List<float>>> kvp in Path_Dict)
        {
            Debug.Log($"Key: {kvp.Key}, Value: {kvp.Value}");
            List<List<float>> path_points = kvp.Value;
            List<Vector3> directions = new List<Vector3>();
            foreach (List<float> point in path_points)
            {
                float pixel_x = point[1];
                float pixel_y = point[0];
                Vector2 pixelCoord = new Vector2(pixel_x, pixel_y);
                Debug.Log("pixelCoord:" + pixelCoord.x.ToString() + pixelCoord.y.ToString());
                Vector3 direction = PixelCoordToWorldCoord(cameraToWorldMatrix, projectionMatrix, pixelCoord);
                Debug.Log("direction:" + direction.x.ToString() + direction.y.ToString() + direction.z.ToString());
                directions.Add(direction);
            }
            Dirs_Dict[kvp.Key] = directions;
        }
        return Dirs_Dict;
    }



    private static Vector3 StringToVector3(string str)
    {
        str = str.Replace("(", " ").Replace(")", " "); //将字符串中"("和")"替换为" "
        string[] s = str.Split(',');
        return new Vector3(float.Parse(s[0]), float.Parse(s[1]), float.Parse(s[2]));
    }

    private static Quaternion StringToQuaternion(string str)
    {
        str = str.Replace("(", " ").Replace(")", " "); //将字符串中"("和")"替换为" "
        string[] s = str.Split(',');
        Quaternion q = new Quaternion();
        q.w = float.Parse(s[0]);
        q.x = float.Parse(s[1]);
        q.y = float.Parse(s[2]);
        q.z = float.Parse(s[3]);
        return q;
    }

    private static Matrix4x4 StringToMatrix(string str)
    {
        str = str.Replace("\n", ",").Replace("\t", ",");
        string[] s = str.Split(',');
        Matrix4x4 cameraToWorldMatix = new Matrix4x4();
        for (int col = 0; col < 4; col++)
        {
            for (int row = 0; row < 4; row++)
            {
                cameraToWorldMatix[row, col] = float.Parse(s[row * 4 + col]);
            }
        }
        // Debug.Log("cameraToWorldMatrix:" + cameraToWorldMatix.ToString("F6"));
        return cameraToWorldMatix;
    }

    private static Dictionary<string, string> JsonToDictionary(string jsonData)
    {
        return JsonConvert.DeserializeObject<Dictionary<string, string>>(jsonData);
    }

    public void ProjectRouteToSpace(string Dir, string png, string file, Color color)
    {
        List<Vector3> dirs = RouteImageToDirections(Dir + "/route_seg/" + png, Dir + "/json/" + file + ".json");
        Vector3 ori = GetHololensCameraPosByMatrix(cameraToWorldMatrix);
        List<Vector3> HitPointSet = new List<Vector3>();
        foreach (Vector3 dir in dirs)
        {
            Vector3 hitpoint = ProjectPointToSpace(ori, dir, color);
            if (hitpoint != new Vector3(0.0f, 0.0f, 0.0f))
            {
                HitPointSet.Add(hitpoint);
            }
        }
        Debug.Log("HitPointSet:" + HitPointSet.ToString());
        SavePointSet(Dir, png, HitPointSet);
    }

    public void ProjectPathToSpace(string Dir, string file, Color color)
    {
        Dictionary<string, List<Vector3>> Dirs_Dict = PathJsonToDirections(Dir + "/path2D_bs/" + file + "_paths2D_bs.json", Dir + "/json/" + file + ".json");
        Vector3 ori = GetHololensCameraPosByMatrix(cameraToWorldMatrix);
        Dictionary<string, List<string>> Path3D_Dict = new Dictionary<string, List<string>>();
        foreach (KeyValuePair<string, List<Vector3>> kvp in Dirs_Dict)
        {
            List<Vector3> dirs = kvp.Value;
            List<string> HitPointSet = new List<string>();
            foreach (Vector3 dir in dirs)
            {
                Vector3 hitpoint = ProjectPointToSpace(ori, dir, color);
                if (hitpoint != new Vector3(0.0f, 0.0f, 0.0f))
                {
                    //Matrix4x4 worldToCameraMatrix = cameraToWorldMatrix.inverse;
                    //Vector4 cameraPoint = worldToCameraMatrix * new Vector4(hitpoint.x, hitpoint.y, hitpoint.z, 1.0f);
                    //Vector3 cameraPointN = cameraPoint / cameraPoint.w;
                    //HitPointSet.Add(cameraPointN.ToString("F6"));
                    HitPointSet.Add(hitpoint.ToString("F6"));
                }
            }
            Debug.Log("HitPointSet:" + HitPointSet.ToString());
            Path3D_Dict[kvp.Key] = HitPointSet;
        }
        SavePath3DSet(Dir, file, Path3D_Dict);
    }

    public void SavePointSet(string Dir, string png, List<Vector3> HitPointSet)
    {
        string jsonString = ConvertVectorListToString(HitPointSet);
        int lengthToRemove = 4;
        int endIndex = png.Length - lengthToRemove > 0 ? png.Length - lengthToRemove : 0;
        string file = png.Substring(0, endIndex);
        string jsonPath = Dir + "/route_space/" + file + ".json";
        File.WriteAllText(jsonPath, jsonString);
        Debug.Log("HitPointSet has been save to a JSON file.");
    }

    public void SavePath3DSet(string Dir, string file, Dictionary<string, List<string>> Path3D_Dict)
    {
        string jsonString = JsonConvert.SerializeObject(Path3D_Dict);
        if (!Directory.Exists(Dir + "/path3D_sp"))
        {
            Directory.CreateDirectory(Dir + "/path3D_sp");
        }
        string jsonPath = Dir + "/path3D_sp/" + file + ".json";
        File.WriteAllText(jsonPath, jsonString);
        Debug.Log("Path3D_Dict has been save to a JSON file.");
    }

    private static string ConvertVectorListToString(List<Vector3> vectorList)
    {
        // 使用StringBuilder来构建结果字符串
        var stringBuilder = new System.Text.StringBuilder();
        stringBuilder.Append("[");

        // 遍历列表中的每个Vector3对象
        for (int i = 0; i < vectorList.Count; i++)
        {
            // 将Vector3对象转换为字符串，并追加到StringBuilder中
            stringBuilder.Append(vectorList[i].ToString("F6"));

            // 如果不是列表中的最后一个元素，追加一个逗号
            if (i < vectorList.Count - 1)
            {
                stringBuilder.Append(", ");
            }
        }

        stringBuilder.Append("]"); // 完成列表字符串的构建
        return stringBuilder.ToString();
    }


}