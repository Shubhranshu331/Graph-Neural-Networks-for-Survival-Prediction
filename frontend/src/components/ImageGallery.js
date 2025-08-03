const images = ['confusion_matrix.png', 'roc_curve.png', 'titanic_knn_graph.png'];

export default function ImageGallery() {
  return (
    <div className="grid grid-cols-2 gap-4 p-4">
      {images.map((img, i) => (
        <img key={i} src={`https://graph-neural-networks-for-survival.onrender.com/image/${img}`} alt={img} className="rounded shadow" />
      ))}
    </div>
  );
}