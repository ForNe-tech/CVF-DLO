using Newtonsoft.Json;
using System;
using System.IO;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class RouteDisplayManager : MonoBehaviour
{
    public static RouteDisplayManager Instance;
    private string totalDir;
    private bool once = true;
    public GameObject Scene;
    private int lengthToRemove = 5;

    public Transform parent;

    private void Awake()
    {
        Instance = this;
    }

    void Start()
    {
        totalDir = @"G:/Research/DLOs Detection/CVF3D-DLO-main/data/LAB_imgs_design_DLO";
        GetScenePose(totalDir + "/ScenePose.json");
        //GetScenePose("F:/CableVis/ScenePose.json");
        
    }

    
    void Update()
    {
        if (once)
        {
            once = false;
            string[] filePaths = Directory.GetFiles(totalDir + "/route3D_select_bs_dense");
            Debug.Log("Update");
            foreach (string filePath in filePaths)
            {
                DisplayRouteToSpace(filePath);
            }

            //string[] filePaths2 = Directory.GetFiles(totalDir + "/LAB_CABIN_CABLES_NEW/route3D_extracted");
            //Debug.Log("Update");
            //foreach (string filePath in filePaths2)
            //{
            //    DisplayRouteToSpace(filePath);
            //}
        }
    }

    

    public void DisplayPathToSpace(string filePath)
    {
        string Name = Path.GetFileName(filePath);
        string[] NamePart = Name.Split('_');
        string file = NamePart[0];
        string fileType = NamePart[1];
        if (fileType[0] == 'p')
        {
            string jsonContent = File.ReadAllText(filePath);
            Dictionary<string, List<List<float>>> path3D_ls = JsonConvert.DeserializeObject<Dictionary<string, List<List<float>>>>(jsonContent);
            foreach (KeyValuePair<string, List<List<float>>> kvp in path3D_ls)
            {
                string Key = kvp.Key;
                GameObject PathFather = new GameObject(file + "_" + Key);
                List<List<float>> path3D = kvp.Value;
                Debug.Log("Key: " + Key + ",Value: " + path3D.ToString());
                Material material = new Material(Shader.Find("Standard"));
                Color randomColor = new Color(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value);
                material.color = randomColor;
                foreach (List<float> path_point in path3D)
                {
                    Vector3 hit_point = new Vector3(path_point[0], path_point[1], path_point[2]);
                    CreateSphere(hit_point, PathFather, material);
                }
            }
        }
    }

    public void DisplayRouteToSpace(string filePath)
    {
        Debug.Log("Display");
        string Name = Path.GetFileName(filePath);
        int endIndex = Name.Length - lengthToRemove > 0 ? Name.Length - lengthToRemove : 0;
        string file = Name.Substring(0, endIndex);
        Debug.Log(file);
        GameObject PathFather = new GameObject(file);
        string jsonContent = File.ReadAllText(filePath);
        List<List<float>> path3D = JsonConvert.DeserializeObject<List<List<float>>>(jsonContent);
        Material material = new Material(Shader.Find("Standard"));
        Color randomColor = new Color(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value);
        material.color = randomColor;
        foreach (List<float> path_point in path3D)
        {
            Vector3 hit_point = new Vector3(path_point[0], path_point[1], path_point[2]);
            CreateSphere(hit_point, PathFather, material);
        }
        string path3DInParent = WorldToParent(path3D);
        SaveCableEst(path3DInParent, file);
    }

    public string WorldToParent(List<List<float>> CableEst)
    {
        List<float[]> PointsInParent = new List<float[]>();
        foreach (List<float> path_point in CableEst)
        {
            Vector3 hit_point = new Vector3(path_point[0], path_point[1], path_point[2]);
            Vector3 parent_point = parent.InverseTransformPoint(hit_point);
            float[] parentArray = ToArray(parent_point);
            PointsInParent.Add(parentArray);
        }
        string PointsInParentStr = JsonConvert.SerializeObject(PointsInParent);
        return PointsInParentStr;
    }

    public void SaveCableEst(string CableEstStr, string label)
    {
        try
        {
            string path = Application.dataPath + "/Model/LAB_CABLE_est/" + label + ".json";
            using (FileStream fs = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.ReadWrite))
            {
                fs.Seek(0, SeekOrigin.Begin);
                fs.SetLength(0);
                using (StreamWriter sw = new StreamWriter(fs, Encoding.UTF8))
                {
                    sw.WriteLine(CableEstStr);
                }
            }
        }
        catch (Exception e)
        {
            print("保存失败！" + e.Message);
        }
    }

    public static void CreateSphere(Vector3 point, GameObject Father, Material material)
    {
        GameObject SpacePoint = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        SpacePoint.GetComponent<Collider>().enabled = false;
        SpacePoint.transform.localScale = new Vector3(0.01f, 0.01f, 0.01f);
        SpacePoint.transform.position = point;
        SpacePoint.transform.SetParent(Father.transform);
        SpacePoint.AddComponent<MeshFilter>();
        SpacePoint.AddComponent<MeshRenderer>();
        SpacePoint.GetComponent<MeshRenderer>().material = material;
    }

    public void GetScenePose(string jsonPath)
    {
        string jsonContent = File.ReadAllText(jsonPath);
        Dictionary<string, string> Scene_Pose = JsonToDictionary(jsonContent);
        Vector3 Scene_P = StringToVector3(Scene_Pose["Scene_P"].ToString());
        Quaternion Scene_Q = StringToQuaternion(Scene_Pose["Scene_Q"].ToString());
        Vector3 Scene_Scale = new Vector3(1.008004f, 1.008004f, 1.008004f);
        Scene.transform.position = Scene_P;
        Scene.transform.rotation = Scene_Q;
    }

    private static Dictionary<string, string> JsonToDictionary(string jsonData)
    {
        return JsonConvert.DeserializeObject<Dictionary<string, string>>(jsonData);
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

    public static float[] ToArray(Vector3 vector)
    {
        return new float[] { vector.x, vector.y, vector.z };
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

    public void GetCameraPose(string jsonPath)
    {
        string jsonContent = File.ReadAllText(jsonPath);
        Dictionary<string, string> Pose_Json = JsonToDictionary(jsonContent);
        Matrix4x4 cameraToWorldMatrix = StringToMatrix(Pose_Json["cameraToWorldMatrix"].ToString());
        Matrix4x4 projectionMatrix = StringToMatrix(Pose_Json["projectionMatrix"].ToString());
        GetHololensCameraPosByMatrix(cameraToWorldMatrix, projectionMatrix);
        UnityCameraCapture(jsonPath);
    }

    public static void GetHololensCameraPosByMatrix(Matrix4x4 cameraToWorldMatrix, Matrix4x4 projectionMatrix)
    {
        // 提取相机的位置
        Vector3 cameraPosition = cameraToWorldMatrix.GetColumn(3);
        Debug.Log("cameraPosition:" + cameraPosition.ToString());

        // 提取相机的旋转
        Quaternion cameraRotation = Quaternion.LookRotation(-cameraToWorldMatrix.GetColumn(2), -cameraToWorldMatrix.GetColumn(1));
        Debug.Log("cameraRotation:" + cameraRotation.ToString());

        Camera.main.transform.position = cameraPosition;
        Camera.main.transform.rotation = cameraRotation;
        // Camera.main.projectionMatrix = projectionMatrix;
    }

    public void UnityCameraCapture(string camPath)
    {
        Camera.main.clearFlags = CameraClearFlags.SolidColor;
        Camera.main.backgroundColor = Color.white;

        RenderTexture renderTexture = new RenderTexture(Screen.width, Screen.height, 24);
        Camera.main.targetTexture = renderTexture;
        Camera.main.Render();

        Texture2D screenshot = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);
        RenderTexture.active = renderTexture;

        screenshot.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        screenshot.Apply();

        // 旋转Texture2D 180度
        Color[] pixels = screenshot.GetPixels();
        for (int y = 0; y < screenshot.height; y++)
        {
            for (int x = 0; x < screenshot.width; x++)
            {
                screenshot.SetPixel(screenshot.width - 1 - x, screenshot.height - 1 - y, pixels[x + y * screenshot.width]);
            }
        }
        screenshot.Apply();

        byte[] bytes = screenshot.EncodeToPNG();

        string Name = Path.GetFileName(camPath);
        int endIndex = Name.Length - lengthToRemove > 0 ? Name.Length - lengthToRemove : 0;
        string file = Name.Substring(0, endIndex);

        string filePath = totalDir + "/route3D_reconst_only_ORIGIN/" + file + ".png";
        File.WriteAllBytes(filePath, bytes);
        RenderTexture.active = null;
        Camera.main.targetTexture = null;
        renderTexture.Release();
        Destroy(renderTexture);
        Destroy(screenshot);

        Debug.Log("Screenshot saved to: " + filePath);
    }
}
