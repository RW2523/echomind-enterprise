import React from "react";

export interface AvatarProps {
  role: "user" | "assistant";
  src?: string | null;
  className?: string;
  size?: number;
}

export const Avatar: React.FC<AvatarProps> = ({ role, src, className = "", size = 80 }) => {
  const [loaded, setLoaded] = React.useState(false);
  const [error, setError] = React.useState(false);

  const showImage = src && loaded && !error;
  const fallbackLetter = role === "user" ? "You" : "AI";

  return (
    <div
      className={`rounded-full flex items-center justify-center text-white/80 font-semibold overflow-hidden bg-white/10 ${className}`}
      style={{ width: size, height: size }}
    >
      {showImage ? (
        <img
          src={src}
          alt=""
          className="w-full h-full object-cover"
        />
      ) : (
        <>
          {src && !error && (
            <img
              src={src}
              alt=""
              className="absolute w-0 h-0 opacity-0"
              onLoad={() => setLoaded(true)}
              onError={() => setError(true)}
            />
          )}
          <span style={{ fontSize: size * 0.35 }}>{fallbackLetter}</span>
        </>
      )}
    </div>
  );
};
