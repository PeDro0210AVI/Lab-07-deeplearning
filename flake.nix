{

  inputs = {
    # nixos-25.11 ships gensim 4.3.3, disabled for python >= 3.12, which
    # conflicts with the (python >= 3.12-only cached) torch build; unstable
    # carries gensim 4.4.0 which supports modern python, so we track it here.
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs =
    {
      nixpkgs,
      ...
    }:
    let
      forAllSystems =
        function:
        nixpkgs.lib.genAttrs
          [
            "x86_64-linux"
            "aarch64-darwin"
          ]
          (
            system:
            let
              pkgs = nixpkgs.legacyPackages.${system};
              python = pkgs.python3;
              pythonPackages =
                ps: with ps; [
                  ipykernel
                  jupyterlab-vim
                  jupyterlab
                  jupyterlab-lsp
                  python-lsp-server

                  numpy
                  pandas
                  matplotlib
                  scipy
                  statsmodels
                  scikit-learn

                  seaborn

                  torch

                  # NLP / embeddings
                  gensim
                  datasets
                  huggingface-hub
                  nltk
                  tqdm
                  ipywidgets
                  pyarrow
                ];
              pythonEnv = python.withPackages pythonPackages;
            in
            function {
              inherit
                pkgs
                pythonEnv
                ;

            }
          );
    in
    {

      devShells = forAllSystems (
        {
          pkgs,
          pythonEnv,
        }:
        {
          default =

            pkgs.mkShell {

              packages = [
                pythonEnv
                pkgs.texliveFull
              ];

            };
        }
      );

    };
}
